# ppo_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class RolloutBuffer:
    """
    存完整输入 + 生成段动作，并保存 gen_start。
    get() 时自动构造 action_mask（只在有效生成步上做统计）
    """
    def __init__(self):
        self.clear()

    def clear(self):
        self.input_ids = []
        self.attention_masks = []
        self.actions = []
        self.old_logprobs = []
        self.rewards = []
        self.values_old = []
        self.dones = []
        self.gen_starts = []

    def add(self, input_ids, attention_mask, actions, old_logprobs, rewards, values_old, dones, gen_start: int):
        self.input_ids.append(input_ids)             # [1, T_full]
        self.attention_masks.append(attention_mask)  # [1, T_full]
        self.actions.append(actions)                 # [1, Tg]
        self.old_logprobs.append(old_logprobs)       # [1]
        self.rewards.append(rewards)                 # [1]
        self.values_old.append(values_old)           # [1]
        self.dones.append(dones)                     # [1]
        self.gen_starts.append(torch.tensor([gen_start], dtype=torch.long, device=input_ids.device))

    @staticmethod
    def _pad_and_stack(tensor_list):
        """
        把形如 [1, Li] 的条目 pad 到同长度并在 batch 维拼起来 → [B, Lmax]
        支持 long/bool/float 等 dtype；bool 将以 False 作为 pad 值。
        """
        assert len(tensor_list) > 0
        device = tensor_list[0].device
        dtype = tensor_list[0].dtype
        max_len = max(t.size(1) for t in tensor_list)
        outs = []
        for t in tensor_list:
            pad_len = max_len - t.size(1)
            if pad_len > 0:
                pad_val = False if dtype == torch.bool else 0
                pad = torch.full((t.size(0), pad_len), pad_val, dtype=dtype, device=device)
                outs.append(torch.cat([t, pad], dim=1))
            else:
                outs.append(t)
        return torch.cat(outs, dim=0)

    def get(self):
        input_ids = self._pad_and_stack(self.input_ids)                # [B, T_full_max]
        attention_masks = self._pad_and_stack(self.attention_masks)    # [B, T_full_max]
        actions = self._pad_and_stack(self.actions)                    # [B, Lg_max]
        old_logprobs = torch.cat(self.old_logprobs, dim=0).view(-1)    # [B]
        rewards = torch.cat(self.rewards, dim=0).view(-1)              # [B]
        values_old = torch.cat(self.values_old, dim=0).view(-1)        # [B]
        dones = torch.cat(self.dones, dim=0).view(-1)                  # [B]
        gen_starts = torch.cat(self.gen_starts, dim=0).view(-1)        # [B]

        # —— 构造 action_mask：每条样本的有效生成步数 = sum(attn) - gen_start —— #
        lengths_full = attention_masks.to(torch.long).sum(dim=1)                # [B]
        B, Lg_max = actions.size()
        eff = (lengths_full - gen_starts).clamp(min=0)                          # [B]
        eff = torch.minimum(eff, torch.full_like(eff, Lg_max))
        idx = torch.arange(Lg_max, device=actions.device).unsqueeze(0).expand(B, Lg_max)
        action_mask = (idx < eff.unsqueeze(1)).to(actions.dtype)                # [B, Lg_max]，1/0

        return input_ids, attention_masks, actions, old_logprobs, rewards, values_old, dones, gen_starts, action_mask


class GPT2ValueHeadModel(nn.Module):
    def __init__(self, gpt2_name='gpt2'):
        super().__init__()
        from transformers import GPT2Model
        self.gpt2 = GPT2Model.from_pretrained(gpt2_name)
        self.value_head = nn.Linear(self.gpt2.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # (B, T_full, H)
        values = self.value_head(last_hidden).squeeze(-1)  # (B, T_full)
        return values


class PPOTrainer:
    def __init__(self, policy_model, value_model, lr=5e-5, clip_eps=0.2, c1=0.5, c2=0.01, max_grad_norm=1.0):
        self.policy_model = policy_model
        self.value_model = value_model
        self.optimizer_policy = Adam(self.policy_model.parameters(), lr=lr)
        self.optimizer_value = Adam(self.value_model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.c1 = c1
        self.c2 = c2
        self.max_grad_norm = max_grad_norm

    def train_step(
        self,
        input_ids, attention_mask, actions,
        old_logprobs, rewards, values_old, dones,
        gen_starts, action_mask
    ):
        """
        序列级 PPO 更新：
        - 序列级优势：A = (R - V_old)
        - new_logp：从 (gen_start-1) 对齐取每步 logits，按 action_mask 累加
        - value：对生成段位置的 value 取均值，做序列级回归
        - 熵：仅在有效生成步统计
        """
        device = input_ids.device
        B, Lg_max = actions.shape

        # ===== 1) 序列级优势（标准化） =====
        with torch.no_grad():
            advantages = (rewards - values_old).detach()  # [B]
            adv_std = advantages.std(unbiased=False) + 1e-8
            advantages = (advantages - advantages.mean()) / adv_std
            returns = rewards.detach()                    # [B]

        # ===== 2) 新策略序列级 logprob（严格对齐 SoftPrompt） =====
        logits_all = self.policy_model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, T_full, V]
        logp_all = F.log_softmax(logits_all, dim=-1)

        maxT = logp_all.size(1)
        start = (gen_starts - 1).clamp_min(0).unsqueeze(1)                  # [B,1]
        idx = torch.arange(Lg_max, device=device).unsqueeze(0).expand(B, Lg_max)
        pos = (start + idx).clamp(max=maxT - 1)                              # [B,Lg_max]

        # 取出与生成步对齐的 [B,Lg_max,V]，再 gather 到 [B,Lg_max]
        pred = logp_all.gather(dim=1, index=pos.unsqueeze(-1).expand(B, Lg_max, logp_all.size(-1)))
        chosen = torch.gather(pred, 2, actions.unsqueeze(-1)).squeeze(-1)    # [B,Lg_max]
        logp_new = (chosen * action_mask).sum(dim=1)                         # [B]

        # ===== 3) token 级熵，仅在有效步统计 =====
        p = pred.exp()
        ent_t = -(p * pred).sum(dim=-1)                                      # [B,Lg_max]
        entropy_seq = (ent_t * action_mask).sum(dim=1) / action_mask.sum(dim=1).clamp_min(1.0)  # [B]

        # ===== 4) 值函数：对生成段 value 取均值，做序列级回归 =====
        values_all = self.value_model(input_ids=input_ids, attention_mask=attention_mask)        # [B, T_full]
        v_pred_steps = values_all.gather(dim=1, index=pos)                                       # [B, Lg_max]
        v_new_seq = (v_pred_steps * action_mask).sum(dim=1) / action_mask.sum(dim=1).clamp_min(1.0)  # [B]

        # ===== 5) PPO-clip =====
        ratio = torch.exp(logp_new - old_logprobs.view(-1))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(v_new_seq, returns)
        loss = policy_loss + self.c1 * value_loss - self.c2 * entropy_seq.mean()

        # ===== 6) 反传与优化 =====
        self.optimizer_policy.zero_grad(set_to_none=True)
        self.optimizer_value.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.max_grad_norm)
        self.optimizer_policy.step()
        self.optimizer_value.step()

        return float(loss.item()), float(policy_loss.item()), float(value_loss.item())
