# run_ppo_training.py
import os
import gc
import json
import argparse
from typing import List, Dict, Any

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from transformers import GPT2LMHeadModel, GPT2Model
from transformers import BertTokenizer

# === 这里改为从 ppo_trainer 导入（也可把 ppo_trainer.py 命名回 train_ppo_rl.py） ===
from ppo_trainer import GPT2ValueHeadModel, PPOTrainer, RolloutBuffer

from discussion_agents import run_pipeline
from evaluation_service import Evaluator
from soft_prompt_embedding import SoftEmbedding
from cost_tracker import CostTracker, cuda_peak_meter
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ====== 工具函数 ======
def decode(input_ids: torch.Tensor, tokenizer: BertTokenizer):
    """仅保留 20 种常见氨基酸字母，并忽略 [PAD]/[SEP] 等特殊符号"""
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    special_tokens = set(
        t for t in [
            getattr(tokenizer, "pad_token", None),
            getattr(tokenizer, "sep_token", None),
            getattr(tokenizer, "eos_token", None),
        ] if t
    )

    decoded = []
    seq_lens = []
    for seq in input_ids:
        tokens = tokenizer.convert_ids_to_tokens(seq)
        chars = []
        for tok in tokens:
            if tok in special_tokens:
                continue
            aa = tok.upper()
            if aa in valid_aas:
                chars.append(aa)
        s = "".join(chars)
        decoded.append(s)
        seq_lens.append(len(s))
    return decoded, seq_lens


def top_k_top_p_filtering(
        logits: torch.FloatTensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    允许对 [B, T, V] 的 logits 做 top-k/p 过滤（对最后一维生效）
    """
    top_p = float(top_p)
    if top_k > 0:
        top_k = min(max(int(top_k), min_tokens_to_keep), logits.size(-1))
        # 过滤掉低于 top-k 阈值的所有 token
        kth = torch.topk(logits, top_k)[0][..., -1, None]
        indices_to_remove = logits < kth
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if 0 < top_p <= 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits


def sample_with_topk_topp(policy_model, tokenizer, batch_size, max_length, top_k, top_p, device):
    """
    - 初始前缀：10 个占位 id [100..109] + tokenizer.encode("")（空串）；再截成 11 长（与旧稿保持一致）
    - 每步：先对整段 logits 过滤，再取最后一步采样
    - EOS 使用 tokenizer.sep_token_id；当且仅当全部样本到 EOS 才停止
    返回：
      input_ids: LongTensor, [B, 11 + steps]
      init_len:  初始前缀长度（固定 11）
    """
    policy_model.eval()

    base_ids = list(range(100, 110))        # [100..109] 共10个 —— SoftEmbedding 会在前向中裁掉这10个，再拼 learned_prompt
    text_ids = tokenizer.encode("")         # 空串
    init_ids = (base_ids + text_ids)[:11]   # 只保留前 11 个
    init_len = 11

    # 构造 [B, 11]，不足位置补 0
    input_tensor = torch.zeros(batch_size, init_len, dtype=torch.long, device=device)
    for idx, tid in enumerate(init_ids):
        input_tensor[:, idx] = tid

    finished = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = policy_model(input_ids=input_tensor)
            logits = outputs.logits  # [B, T, V]
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            last_logits = logits[:, -1, :]                      # [B, V]
            probs = torch.softmax(last_logits, dim=-1)          # [B, V]
            next_id = torch.multinomial(probs, num_samples=1)   # [B, 1]

            eos_hit = (next_id == tokenizer.sep_token_id)
            finished |= eos_hit
            input_tensor = torch.cat([input_tensor, next_id], dim=1)

            if finished.all():
                break
    return input_tensor, init_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="model_ckpt")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--logdir', type=str, default="output/runs/ppo")
    parser.add_argument('--save_dir', type=str, default="output/ppo_ckpt")
    parser.add_argument('--memory_dir', type=str, default="outputs/memory_dir")
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--response_id', type=int, default=1)
    parser.add_argument('--warmup_dir', type=str, default="")
    parser.add_argument('--initialize_model', action="store_true")
    parser.add_argument('--use_default_reward', action="store_true")
    parser.add_argument('--real_model_train', action="store_true")
    parser.add_argument('--reward_id', type=str, default="rw_default")
    parser.add_argument('--stage_id', type=str, default="i")
    parser.add_argument('--amp_generator_root', type=str, default="")
    parser.add_argument('--amp_regression_root', type=str, default="")
    parser.add_argument('--default_sign',type=int, default=-1)
    parser.add_argument('--workspace_dir', type=str, default="")
    parser.add_argument('--drift_wordlist',type=str, default='')
    # parser.add_argument('--skip_independent_reviewer',type=lambda s: [x.strip() for x in s.split(',') if x.strip()],default=[])
    parser.add_argument(
        '--skip_independent_reviewer',
        nargs='?',                 # 允许 0 或 1 个值
        const='',                  # 不给值时传入空字符串
        type=lambda s: [x.strip() for x in s.split(',') if x.strip()],
        default=[]
    )
    parser.add_argument(
        '--ablation_mode',
        type=int,
        default=0,
        help='0: full; 1: drop Vb; 2: drop Va; 3: drop Sb; 4: drop Va+Vb; 5: drop Vb+Sb; 6: drop Va+Sb; 7: drop Va+Vb+Sb'
    )
    # 新增：实验与本地模型/服务配置
    parser.add_argument(
        '--experiment',
        type=str,
        default=os.getenv("EXPERIMENT", "baseline_api"),
        choices=[
            "baseline_api",
            "all_llama_local",
            "all_qwen_local",
            "qwen_reviewers_gpt5_ac",
            "api_reviewers_qwen_ac",
        ],
        help="选择五种实验形态之一"
    )
    parser.add_argument(
        '--ollama_host',
        type=str,
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama 服务地址"
    )
    parser.add_argument(
        '--llama_tag',
        type=str,
        default=os.getenv("OLLAMA_LLAMA_TAG", "llama3.1:8b"),
        help="Ollama 本地 Llama 模型标签"
    )
    parser.add_argument(
        '--qwen_tag',
        type=str,
        default=os.getenv("OLLAMA_QWEN_TAG", "qwen2.5:7b-instruct"),
        help="Ollama 本地 Qwen 模型标签"
    )


    args = parser.parse_args()
    # tracker = CostTracker.get_global()
    # gpu_count = 1  # 如果你后续支持多卡，可动态检测或从参数传入

    # evaluator = Evaluator(args.use_default_reward,amp_generator_root=args.amp_generator_root, mic_regression_root=args.amp_regression_root, default_sign=args.default_sign,workspace_dir=args.workspace_dir)
    # if len(args.skip_independent_reviewer)==3:
    #     evaluator = Evaluator(True,amp_generator_root=args.amp_generator_root, mic_regression_root=args.amp_regression_root, default_sign=args.default_sign,workspace_dir=args.workspace_dir)
    
    tracker = CostTracker.get_global()
    gpu_count = 1  # 如果你后续支持多卡，可动态检测或从参数传入

    # Sb 消融（3/5/6/7）时：强制使用内部手工 reward（非 LLM 生成的 RL 函数）
    use_default_reward = args.use_default_reward or (args.ablation_mode in [3, 5, 6, 7])

    evaluator = Evaluator(
        use_default_reward,
        amp_generator_root=args.amp_generator_root,
        mic_regression_root=args.amp_regression_root,
        default_sign=args.default_sign,
        workspace_dir=args.workspace_dir,
        ablation_mode=args.ablation_mode,
    )

    # 跳过 independent reviewer 的逻辑仍然保留，但也要带上 ablation_mode
    if len(args.skip_independent_reviewer) == 3:
        evaluator = Evaluator(
            True,
            amp_generator_root=args.amp_generator_root,
            mic_regression_root=args.amp_regression_root,
            default_sign=args.default_sign,
            workspace_dir=args.workspace_dir,
            ablation_mode=args.ablation_mode,
        )

    
    memory_log_path = f"{args.memory_dir}/iter_{args.iter}/{args.reward_id}_memory_log.txt"
    if args.real_model_train:
        memory_log_path = "real_model_" + memory_log_path
    os.makedirs(os.path.dirname(memory_log_path), exist_ok=True)

    print("==== PPO Peptide Training Script Started ====")
    print("fps step: PPO training loop started", flush=True)
    print(f"Tensorboard Directory: {args.logdir}", flush=True)
    print(f'Default sign: ', str(args.default_sign))
    print(f"Max_length: ", str(args.max_length))
    print(f"workspace_dir:", args.workspace_dir)
    print(f"ablation_mode:", args.ablation_mode)
    print(f"use_default_reward:", str(use_default_reward))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=args.logdir)

    # === 统一 tokenizer（与旧工程保持一致） ===
    args.vocab_path = f'{args.amp_generator_root}/voc/vocab.txt'
    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    eos_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    # ====== 初始化 / 加载 ======
    if args.initialize_model:
        prompt_model_load = torch.load(f"{args.model_path}/pytorch_model.bin", map_location=device)
        policy_model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)

        s_wte = SoftEmbedding(policy_model.get_input_embeddings(), n_tokens=10, initialize_from_vocab=True)
        with torch.no_grad():
            s_wte.learned_embedding.data.copy_(prompt_model_load['transformer.wte.learned_embedding'].to(device))
            s_wte.wte.weight.data.copy_(prompt_model_load['transformer.wte.wte.weight'].to(device))
        s_wte = s_wte.to(device)

        policy_model.set_input_embeddings(s_wte)
        policy_model.to(device)  # ← 注入后再搬一次，确保万无一失
        
        value_model = GPT2ValueHeadModel("gpt2").to(device)

        # 保存一次“latest”快照
        os.makedirs(args.save_dir, exist_ok=True)
        tokenizer.save_pretrained(os.path.join(args.save_dir, "tokenizer"))
        policy_model.save_pretrained(os.path.join(args.save_dir, "latest_policy"))
        value_model.gpt2.save_pretrained(os.path.join(args.save_dir, "latest_value_gpt"))
        torch.save(value_model.value_head.state_dict(), os.path.join(args.save_dir, "latest_value_head.pt"))

        # 可选预热（略）
        if args.warmup_dir and os.path.exists(args.warmup_dir):
            warmup_df = pd.read_csv(args.warmup_dir)
            sequences = warmup_df["sequence"].tolist()[:3]

            # 评估 & 记录（与旧稿一致）
            all_decoded = [s for s in sequences if len(s) > 1]
            evaluation_result, physchem, toxicity, amp_act, structure, similarity, mic_score, mic_original = evaluator.get_evaluation_outputs_from_agent(all_decoded)
            peptides_info = evaluator.parse_batch_evaluation(evaluation_result)
            meta_base = {"stage": str(args.stage_id), "iter": int(args.iter), "response_id": int(getattr(args, "response_id", 0)), "epoch": 0}
            # critical_comments, llm_overall_score, _ = run_pipeline(peptides_info, args.amp_generator_root, args.skip_independent_reviewer,meta=meta_base, drift_wordlist=args.drift_wordlist)

            critical_comments, llm_overall_score, _ = run_pipeline(
                peptides_info,
                args.amp_generator_root,
                args.skip_independent_reviewer,
                meta=meta_base,
                drift_wordlist=args.drift_wordlist,
                experiment=args.experiment,  # ★ 新增
                local_models={"llama": args.llama_tag, "qwen": args.qwen_tag},  # ★ 新增
                ollama_host=args.ollama_host  # ★ 新增
            )

            amp_scores = [p.get("amp_score", 0.0) for p in peptides_info]
            mic_scores = [p.get("mic_score", 0.0) for p in peptides_info]
            mic_raw    = [p.get("mic_original", 0.0) for p in peptides_info]

            avg_amp_score = float(np.mean(amp_scores)) if amp_scores else 0.0
            avg_mic_score = float(np.mean(mic_scores)) if mic_scores else 0.0
            avg_mic_raw   = float(np.mean(mic_raw)) if mic_raw else 0.0

            cur_epoch_log = f"[Stage: {args.stage_id}][Default:Real AMP Result][Raw MIC Value: {avg_mic_raw}, MIC Score: {avg_mic_score}][AMP Score: {avg_amp_score}][Meta Review Score: {llm_overall_score}][Meta Review Message:{critical_comments}]"
            with open(memory_log_path, "a", encoding="utf-8") as f:
                f.write(cur_epoch_log + "\n")
        out_path = os.path.join(args.save_dir, "compute_costs_local.json")
        try:
            tracker.dump_json(out_path)
            print(f"[CostTracker] dumped to: {out_path}")
        except Exception as e:
            print(f"[CostTracker] dump failed: {e}")
        print("Initialization finished.")
        writer.close()
        return

    if not args.real_model_train:
        args.save_dir = os.path.join(args.save_dir, args.reward_id)
    
    best_policy_dir = os.path.join(args.model_path, "best_policy")
    last_policy_dir = os.path.join(args.model_path, "latest_policy")

    best_policy_ckpt = os.path.join(best_policy_dir, "pytorch_model.bin")
    last_policy_ckpt = os.path.join(last_policy_dir, "pytorch_model.bin")

    if os.path.isdir(best_policy_dir) and os.path.isfile(best_policy_ckpt):
        policy_dir = best_policy_dir
        policy_ckpt = best_policy_ckpt
        print(f"[Policy] Loading BEST policy from {policy_dir}")
    elif os.path.isdir(last_policy_dir) and os.path.isfile(last_policy_ckpt):
        policy_dir = last_policy_dir
        policy_ckpt = last_policy_ckpt
        print(f"[Policy] BEST policy not found, loading LAST policy from {policy_dir}")
    else:
        raise FileNotFoundError(
            f"Neither best_policy nor last_policy checkpoints found under {args.model_path}"
        )

    policy_model = GPT2LMHeadModel.from_pretrained(policy_dir).to(device)
    soft_prompt_ckpt = torch.load(policy_ckpt, map_location=device)

    s_wte = SoftEmbedding(policy_model.get_input_embeddings(), n_tokens=10, initialize_from_vocab=True)
    with torch.no_grad():
        s_wte.learned_embedding.data.copy_(soft_prompt_ckpt['transformer.wte.learned_embedding'].to(device))
        s_wte.wte.weight.data.copy_(soft_prompt_ckpt['transformer.wte.wte.weight'].to(device))
    s_wte = s_wte.to(device)

    policy_model.set_input_embeddings(s_wte)
    policy_model.to(device)  # ← 再搬一次，确保 embedding/soft_prompt 都在 cuda
    policy_model.config.eos_token_id = eos_id
    policy_model.config.pad_token_id = pad_id
    value_model = GPT2ValueHeadModel("gpt2").to(device)
    best_value_gpt_dir = os.path.join(args.model_path, "best_value_gpt")
    last_value_gpt_dir = os.path.join(args.model_path, "latest_value_gpt")

    best_value_head_path = os.path.join(args.model_path, "best_value_head.pt")
    last_value_head_path = os.path.join(args.model_path, "latest_value_head.pt")

    if os.path.isdir(best_value_gpt_dir) and os.path.isfile(best_value_head_path):
        value_gpt_dir = best_value_gpt_dir
        value_head_path = best_value_head_path
        print(f"[Value] Loading BEST value model from {value_gpt_dir}")
    elif os.path.isdir(last_value_gpt_dir) and os.path.isfile(last_value_head_path):
        value_gpt_dir = last_value_gpt_dir
        value_head_path = last_value_head_path
        print(f"[Value] BEST value model not found, loading LAST value model from {value_gpt_dir}")
    else:
        raise FileNotFoundError(
            f"Neither best_value_* nor last_value_* checkpoints found under {args.model_path}"
        )

    value_model.gpt2 = GPT2Model.from_pretrained(value_gpt_dir).to(device)
    value_model.value_head.load_state_dict(torch.load(value_head_path, map_location="cpu"))
    value_model.value_head.to(device)

    trainer = PPOTrainer(policy_model, value_model)
    buffer = RolloutBuffer()
    best_reward = -float("inf")

    for epoch in range(args.epochs):
        with cuda_peak_meter() as mtr:
            epoch_t0 = time.time()
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            buffer.clear()
            all_rewards: List[float] = []
            all_decoded: List[str] = []
            all_input_ids = []
            all_attention_masks = []
            all_actions = []
            all_logprobs = []

            # === 并行采样（严格与 predict 一致的生成方式） ===
            input_ids_batch, init_len = sample_with_topk_topp(
                policy_model=policy_model,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                top_k=50,
                top_p=0.95,
                device=device,
            )  # [B, T], init_len=11

            # 注意：attention_mask 用 long（0/1），对 Transformers 更通用
            attention_mask_batch = torch.ones_like(input_ids_batch, dtype=torch.long, device=input_ids_batch.device)

            # 只把“生成的 token 段”当作 actions（去掉前缀 init_len 个）
            actions_batch = input_ids_batch[:, init_len:].clone()  # [B, Tg]
            if actions_batch.numel() == 0:
                continue

            # old_logp（严格对齐：第一个生成 token 由位置 (init_len-1) 的 logits 预测）
            with torch.no_grad():
                logits_batch = policy_model(input_ids=input_ids_batch, attention_mask=attention_mask_batch).logits  # [B, T, V]
                log_probs_batch = torch.log_softmax(logits_batch, dim=-1)                                          # [B, T, V]
                V = logits_batch.size(-1)
                if (actions_batch >= V).any() or (actions_batch < 0).any():
                    raise ValueError(f"actions contain OOV ids (vocab={V}). Check init_len slicing.")
                pred_slice = log_probs_batch[:, init_len-1:-1, :]                                                   # [B, Tg, V]
                chosen_log_probs_batch = torch.gather(pred_slice, 2, actions_batch.unsqueeze(-1)).squeeze(-1)      # [B, Tg]
                logprobs_batch = chosen_log_probs_batch.sum(dim=1)                                                  # [B]

            # 解码只针对“生成段”

            decoded_batch, _ = decode(actions_batch.cpu(), tokenizer)
            keep_mask = [len(s) > 1 for s in decoded_batch]

            # 收集有效样本（按样本逐一对齐）
            for b in range(args.batch_size):
                if not keep_mask[b]:
                    continue
                all_decoded.append(decoded_batch[b])
                all_input_ids.append(input_ids_batch[b:b+1])
                all_attention_masks.append(attention_mask_batch[b:b+1])
                all_actions.append(actions_batch[b:b+1])
                all_logprobs.append(logprobs_batch[b:b+1])

            # === 评估 & 逐样本奖励（保持你改后的“逐样本”逻辑） ===
            if not all_decoded:
                continue

            gen_tokens = 0
            for a in all_actions:
                gen_tokens += int(a.numel())  # a: [1, Tg]，numel() 即生成 token 数
            samples = len(all_decoded)
            # 评估 & 逐样本奖励（不变） —— 这里的 run_pipeline 也透传 meta
            meta_base = {"stage": str(args.stage_id), "iter": int(args.iter), "response_id": int(args.response_id), "epoch": int(epoch)}
            evaluation_result, physchem, toxicity, amp_act, structure, similarity, mic_score, mic_original = evaluator.get_evaluation_outputs_from_agent(all_decoded)
            peptides_info = evaluator.parse_batch_evaluation(evaluation_result)
            # critical_comments, llm_overall_score, _ = run_pipeline(peptides_info, args.amp_generator_root, args.skip_independent_reviewer,meta=meta_base,drift_wordlist=args.drift_wordlist)
            critical_comments, llm_overall_score, _ = run_pipeline(
                peptides_info,
                args.amp_generator_root,
                args.skip_independent_reviewer,
                meta=meta_base,
                drift_wordlist=args.drift_wordlist,
                experiment=args.experiment,  # ★ 新增
                local_models={"llama": args.llama_tag, "qwen": args.qwen_tag},  # ★ 新增
                ollama_host=args.ollama_host  # ★ 新增
            )

            amp_scores = [p.get("amp_score", 0.0) for p in peptides_info]
            mic_scores = [p.get("mic_score", 0.0) for p in peptides_info]
            mic_raw    = [p.get("mic_original", 0.0) for p in peptides_info]

            avg_amp_score = float(np.mean(amp_scores)) if amp_scores else 0.0
            avg_mic_score = float(np.mean(mic_scores)) if mic_scores else 0.0
            avg_mic_raw   = float(np.mean(mic_raw)) if mic_raw else 0.0

            # ✅ 修复后的逐样本奖励：不再“批均值→广播”
            rewards_list = [
                evaluator.compute_rewards(mic,amp, llm_overall_score)
                for mic,amp in zip(mic_scores,amp_scores)
            ]
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)  # [B_valid]

            cur_epoch_log = (
                f"[Stage: {args.stage_id}][Epoch: {epoch}]"
                f"[Rewards:{rewards_list}]"
                f"[Raw MIC Value: {avg_mic_raw}, MIC Score: {avg_mic_score}]"
                f"[AMP Score: {avg_amp_score}]"
                f"[Meta Review Score: {llm_overall_score}]"
                f"[Meta Review Message:{critical_comments}]"
            )
            with open(memory_log_path, "a", encoding="utf-8") as f:
                f.write(cur_epoch_log + "\n")

            # === 填充 Buffer：注意传“完整输入 + gen_start”，由 trainer 内部做对齐与 mask ===
            for b in range(len(all_decoded)):
                input_ids = all_input_ids[b]            # [1, T_full]
                attention_mask = all_attention_masks[b] # [1, T_full]
                actions = all_actions[b]                # [1, Tg]
                old_logp = all_logprobs[b]              # [1]
                reward = rewards[b].unsqueeze(0)        # [1]

                with torch.no_grad():
                    v_all = value_model(input_ids=input_ids, attention_mask=attention_mask)  # [1, T_full] or [1, T_full, 1]
                    if v_all.dim() == 3:
                        v_all = v_all.squeeze(-1)
                    # 用与 old_logp 对齐的生成段位置 [init_len-1 : -1] 做均值，得到序列级 V_old
                    V_old_seq = v_all[:, init_len-1:-1].mean(dim=1)  # [1]

                dones = torch.ones_like(reward)
                # 关键修复：不要把 input_ids 切成片段；传完整 + gen_start
                buffer.add(
                    input_ids,                # [1, T_full]
                    attention_mask,           # [1, T_full]
                    actions,                  # [1, Tg]
                    old_logp,                 # [1]
                    reward,                   # [1]
                    V_old_seq,                # [1]
                    dones,                    # [1]
                    gen_start=init_len        # ★★ 新增：用于 trainer 内部对齐
                )
                all_rewards.append(float(reward.item()))

            # === PPO 更新 ===
            if len(all_rewards) > 0:
                batch = buffer.get()
                loss, policy_loss, value_loss = trainer.train_step(*batch)

                avg_reward = float(np.mean(all_rewards))

                # 简单 success rate（与你原逻辑一致）
                success_threshold = np.percentile(all_rewards, 80) if all_rewards else 0.0
                success_count = sum([1 for r in all_rewards if r > success_threshold])
                success_rate = success_count / len(all_rewards) if all_rewards else 0.0

                writer.add_scalar("Loss/total", loss, epoch)
                writer.add_scalar("Loss/policy", policy_loss, epoch)
                writer.add_scalar("Loss/value", value_loss, epoch)
                writer.add_scalar("Reward/avg", avg_reward, epoch)
                writer.add_scalar("Reward/success_rate", success_rate, epoch)

                print("Average reward:", avg_reward)

                os.makedirs(args.save_dir, exist_ok=True)
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    print("Saving best model (reward = {:.4f})".format(best_reward))
                    policy_model.save_pretrained(os.path.join(args.save_dir, "best_policy"))
                    value_model.gpt2.save_pretrained(os.path.join(args.save_dir, "best_value_gpt"))
                    torch.save(value_model.value_head.state_dict(), os.path.join(args.save_dir, "best_value_head.pt"))

                epoch_dir = os.path.join(args.save_dir, f"{epoch:04d}")

                # 先确保顶层和本轮目录都存在
                os.makedirs(epoch_dir, exist_ok=True)

                # 然后保存
                tokenizer.save_pretrained(os.path.join(epoch_dir, "tokenizer"))
                policy_model.save_pretrained(os.path.join(epoch_dir, "latest_policy"))
                value_model.gpt2.save_pretrained(os.path.join(epoch_dir, "latest_value_gpt"))
                torch.save(value_model.value_head.state_dict(),os.path.join(epoch_dir, "latest_value_head.pt"))

                tokenizer.save_pretrained(os.path.join(args.save_dir, "tokenizer"))
                policy_model.save_pretrained(os.path.join(args.save_dir, "latest_policy"))
                value_model.gpt2.save_pretrained(os.path.join(args.save_dir, "latest_value_gpt"))
                torch.save(value_model.value_head.state_dict(), os.path.join(args.save_dir,"latest_value_head.pt"))

                torch.cuda.empty_cache()
                gc.collect()
            duration = time.time() - epoch_t0
            peak_mb = mtr.peak_mb  # torch 记录到的本进程峰值显存
            tracker.log_train_epoch(
                name="ppo_epoch",
                stage=str(args.stage_id),
                iter_idx=int(args.iter),
                epoch=int(epoch),
                response_id=int(args.response_id),
                duration_sec=duration,
                gpu_count=gpu_count,
                peak_mem_mb=peak_mb,
                generated_tokens=gen_tokens,
                samples=samples,
            )
    out_path = os.path.join(args.save_dir, "compute_costs_local.json")
    try:
        tracker.dump_json(out_path)
        print(f"[CostTracker] dumped to: {out_path}")
    except Exception as e:
        print(f"[CostTracker] dump failed: {e}")
    print("fps step: training completed", flush=True)
    writer.close()


if __name__ == '__main__':
    main()
