# -*- coding: utf-8 -*-
import os
import re
import time
import json
import torch
import logging
import requests
from typing import List, Dict, Optional, Any
from collections import OrderedDict

# ===== 远端 LLM SDK =====
import openai
import google.generativeai as genai

# ===== 你项目已有工具 =====
from utils.perplexity_helper import replace_scores_in_perplexity
from utils.area_chair_calculate_helper import format_area_meta_with_scores
from utils.utils import file_to_string
from cost_tracker import CostTracker, log_openai_completion, log_perplexity_completion, log_gemini_response

logger = logging.getLogger(__name__)

# =============================================================================
# 配置区：API Key / Ollama
# =============================================================================
# 建议通过环境变量配置 API Key，避免硬编码
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "Your API KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "Your API KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", 'Your API KEY')

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Ollama 服务（容器内无 systemd；请确保已执行 `ollama serve &`）
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# =============================================================================
# 辅助函数
# =============================================================================
def _extract_text(resp) -> str:
    """从 Gemini SDK 返回结果中提取纯文本"""
    if not resp or not getattr(resp, "candidates", None):
        return ""
    texts = []
    for c in resp.candidates:
        if hasattr(c, "content") and getattr(c.content, "parts", None):
            for p in c.content.parts:
                t = getattr(p, "text", None)
                if t:
                    texts.append(t)
    return "\n".join(texts).strip()

# =============================================================================
# LLMWrapper：统一封装 openai/perplexity/genai/ollama
# =============================================================================
class LLMWrapper:
    """
    name: 'openai' | 'perplexity' | 'genai' | 'ollama'
    - 远端：走各自官方 API
    - 本地：通过 Ollama HTTP Chat 接口调用（/api/chat）
    """
    def __init__(
        self,
        name: str,
        system_prompt: str,
        max_tokens: int,
        amp_generator_root: str,
        # 仅 Ollama 分支使用：
        local_ollama_model: Optional[str] = None,
        ollama_host: Optional[str] = None,
        local_options: Optional[Dict[str, Any]] = None
    ):
        self.name = name.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.amp_generator_root = amp_generator_root

        # for Ollama
        self.local_ollama_model = local_ollama_model  # e.g. 'llama3.1:8b' / 'qwen2.5:7b-instruct'
        self.ollama_host = ollama_host or OLLAMA_HOST
        self.local_options = local_options or {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": max_tokens or 512,
        }

    def generate_response(self, prompt: str, meta: Optional[Dict[str, Any]] = None) -> str:
        if self.name == "openai":
            return self.call_openai(prompt, meta=meta)
        elif self.name == "perplexity":
            return self.call_perplexity(prompt, meta=meta)
        elif self.name == "genai":
            return self.call_genai(prompt, meta=meta)
        elif self.name in ("ollama", "ollama-llama", "ollama-qwen"):
            return self.call_ollama(prompt, meta=meta)
        else:
            raise ValueError(f"Unknown model: {self.name}")

    # ---------------- Ollama（本地）----------------
    def call_ollama(self, prompt: str, meta: Optional[Dict[str, Any]] = None) -> str:
        model = self.local_ollama_model or os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.1:8b")
        url = f"{self.ollama_host}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system_prompt or "You are a helpful assistant."},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": float(self.local_options.get("temperature", 0.7)),
                "top_p": float(self.local_options.get("top_p", 0.9)),
                # Ollama 用 num_predict 控制生成长度
                "num_predict": int(self.local_options.get("num_predict", self.max_tokens or 512)),
            }
        }
        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, timeout=600)
                r.raise_for_status()
                data = r.json()
                content = ""
                if isinstance(data, dict) and "message" in data:
                    content = data["message"].get("content", "") or ""
                if content.strip():
                    return content.strip()
                logger.warning(f"[Ollama] Attempt {attempt+1} returned empty content. Retrying...")
                time.sleep(2)
            except Exception as e:
                logger.error(f"[Ollama] Attempt {attempt+1} failed: {e}")
                time.sleep(2)
        return "Local (Ollama) model did not provide a comment"

    # ---------------- Google GenAI ----------------
    def call_genai(self, prompt: str, meta: Optional[Dict[str, Any]] = None) -> str:
        if not GOOGLE_API_KEY:
            logger.error("[GenAI] GOOGLE_API_KEY 未设置")
            return "GenAI did not provide a comment (missing GOOGLE_API_KEY)"
        model = genai.GenerativeModel(model_name="gemini-2.5-pro", system_instruction=self.system_prompt)
        for attempt in range(3):
            try:
                response = model.generate_content(contents=prompt)
                text = _extract_text(response)

                info = meta or {}
                try:
                    log_gemini_response(
                        response,
                        agent=info.get("agent", "unknown"),
                        stage=info.get("stage"),
                        epoch=info.get("epoch"), iter_idx=info.get("iter"),
                        response_id=info.get("response_id"),
                    )
                except Exception:
                    pass

                if text:
                    return text
                logger.warning(f"[GenAI] Attempt {attempt+1} returned no text. Retrying in 10s...")
                time.sleep(10)
            except Exception as e:
                logger.error(f"[GenAI] Attempt {attempt+1} failed with error: {e}")
                time.sleep(10)
        return "GenAI did not provide a comment"

    # ---------------- OpenAI ----------------
    def call_openai(self, prompt: str, meta: Optional[Dict[str, Any]] = None) -> str:
        if not OPENAI_API_KEY or openai_client is None:
            logger.error("[OpenAI] OPENAI_API_KEY 未设置")
            return "OpenAI did not provide a comment (missing OPENAI_API_KEY)"
        for attempt in range(3):
            try:
                resp = openai_client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )

                info = meta or {}
                try:
                    log_openai_completion(
                        resp,
                        agent=info.get("agent", "unknown"),
                        stage=info.get("stage"),
                        epoch=info.get("epoch"), iter_idx=info.get("iter"),
                        response_id=info.get("response_id"),
                    )
                except Exception:
                    pass

                content = None
                if resp and getattr(resp, "choices", None):
                    msg = getattr(resp.choices[0], "message", None)
                    if msg:
                        content = getattr(msg, "content", None)
                if content and content.strip():
                    return content.strip()
                logger.warning(f"[OpenAI] Attempt {attempt+1} returned no text. Retrying in 10s...")
                time.sleep(10)
            except Exception as e:
                logger.error(f"[OpenAI] Attempt {attempt+1} failed with error: {e}")
                time.sleep(10)
        return "OpenAI did not provide a comment"

    # ---------------- Perplexity ----------------
    def call_perplexity(self, prompt: str, meta: Optional[Dict[str, Any]] = None) -> str:
        if not PERPLEXITY_API_KEY:
            logger.error("[Perplexity] PERPLEXITY_API_KEY 未设置")
            return "Perplexity did not provide a comment (missing PERPLEXITY_API_KEY)"
        url = "https://api.perplexity.ai/chat/completions"
        headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
        }
        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()
                content = None
                try:
                    j = response.json()
                    if j and "choices" in j and j["choices"]:
                        msg = j["choices"][0].get("message", {})
                        content = msg.get("content")

                    info = meta or {}
                    try:
                        log_perplexity_completion(
                            j,
                            agent=info.get("agent", "unknown"),
                            stage=info.get("stage"),
                            epoch=info.get("epoch"), iter_idx=info.get("iter"),
                            response_id=info.get("response_id"),
                        )
                    except Exception:
                        pass
                except Exception as json_err:
                    logger.error(f"[Perplexity] JSON parse error on attempt {attempt+1}: {json_err}")

                if content and (text := content.strip()):
                    fixed, _results = replace_scores_in_perplexity(text, amp_generator_root=self.amp_generator_root)
                    return (fixed or "").strip()

                logger.warning(f"[Perplexity] Attempt {attempt+1} returned no text. Retrying in 10s...")
                time.sleep(10)
            except Exception as e:
                sc = response.status_code if 'response' in locals() else "N/A"
                tx = response.text if 'response' in locals() else "N/A"
                logger.error(f"[Perplexity] Attempt {attempt+1} failed (status={sc}): {e} | body={tx}")
                time.sleep(10)
        return "Perplexity did not provide a comment"

# =============================================================================
# Agents
# =============================================================================
class ExpertAgent:
    def __init__(self, name: str, model: LLMWrapper):
        self.name = name
        self.model = model
    def generate_response(self, prompt: str, meta: Optional[Dict[str, Any]] = None) -> str:
        return self.model.generate_response(prompt, meta=meta)

class MetaReviewAgent:
    def __init__(self, model: LLMWrapper):
        self.model = model
    def synthesize_and_score(self, evaluations: Dict[str, str], meta: Optional[Dict[str, Any]] = None) -> str:
        prompt = ""
        for agent, opinion in evaluations.items():
            prompt += f"[{agent}]：{opinion}\n"
        return self.model.generate_response(prompt, meta=meta)

# =============================================================================
# 数据与 Prompt
# =============================================================================
def build_peptide_json(peptides_info):
    def _to_number(x):
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return None

    def _prop_to_string(v):
        if v is None:
            return ""
        if isinstance(v, dict):
            return "; ".join(f"{k}={v[k]}" for k in v)
        if isinstance(v, (list, tuple, set)):
            return "; ".join(map(str, v))
        return str(v)

    n = len(peptides_info)
    out = {
        "id": [f"X{i+1}" for i in range(n)],
        "seq": [],
        "amp_score": [],
        "mic_value": [],
        "plddt_score": [],
        "toxicity_score": [],
        "sim_score": [],
        "prop": [],
    }
    for p in peptides_info:
        out["seq"].append(p.get("sequence"))
        out["amp_score"].append(_to_number(p.get("amp_score")))
        out["mic_value"].append(_to_number(p.get("mic_original")))
        out["plddt_score"].append(_to_number(p.get("plddt_score")))
        out["toxicity_score"].append(_to_number(p.get("toxicity_score")))
        out["sim_score"].append(_to_number(p.get("similarity_score")))
        out["prop"].append(_prop_to_string(p.get("physicochemical_propertie")))

    json_str = json.dumps(out, ensure_ascii=False)
    gemini_content = [{"role": "user", "parts": [{"text": json_str}]}]
    return out, json_str, gemini_content

def _load_prompt(amp_generator_root, drift=''):
    if drift == 'plus0.1':
        INDEPENDENT_REVIEWER_AGENT = file_to_string(
            f"{amp_generator_root}/utils/prompt/INDEPENDENT_REVIEWER_AGENT_PLUS.txt"
        )
    elif drift == 'minus0.1':
        INDEPENDENT_REVIEWER_AGENT = file_to_string(
            f"{amp_generator_root}/utils/prompt/INDEPENDENT_REVIEWER_AGENT_MINUS.txt"
        )
    else:
        INDEPENDENT_REVIEWER_AGENT = file_to_string(
            f"{amp_generator_root}/utils/prompt/INDEPENDENT_REVIEWER_AGENT.txt"
        )
    AREA_CHAIR_AGENT = file_to_string(f"{amp_generator_root}/utils/prompt/AREA_CHAIR_AGENT.txt")
    return INDEPENDENT_REVIEWER_AGENT, AREA_CHAIR_AGENT

def _load_default_empty_evaluation():
    return '''[EFF:][comment]None[Tags: None][notes]None[Dist: (0)] [Num: 1]
[Safe:][comment]None[Tags: None][notes]None[Dist: (0)] [Num: 1]
[DevStruct:][comment]None[Tags: None][notes]None[Dist: (0)] [Num: 1]
[Orig:][comment]None[Tags: None][notes]None[Dist: (0)] [Num: 1]
'''

def _make_local_triplet(base_label: str, make_wrapper_fn):
    """
    生成本地 reviewer 三件套：<base_label> 1/2/3
    例如 base_label='Llama' -> {'Llama 1': ExpertAgent(...), ...}
    三位 reviewer 共享同一 LLMWrapper（避免重复加载模型）。
    """
    shared = make_wrapper_fn()
    return OrderedDict([
        (f"{base_label} 1", ExpertAgent(f"{base_label} 1", shared)),
        (f"{base_label} 2", ExpertAgent(f"{base_label} 2", shared)),
        (f"{base_label} 3", ExpertAgent(f"{base_label} 3", shared)),
    ])

# =============================================================================
# 构建 5 种实验的 Reviewer/AC
# =============================================================================
def build_agents_by_experiment(
    experiment: str,
    INDEPENDENT_REVIEWER_AGENT: str,
    AREA_CHAIR_AGENT: str,
    amp_generator_root: str,
    local_models: Optional[Dict[str, str]] = None,
    ollama_host: Optional[str] = None,
    local_options: Optional[Dict[str, Any]] = None
):
    """
    返回: (agents_dict, meta_reviewer_agent)

    - baseline_api / api_reviewers_qwen_ac:
        reviewer 名称为: GenAI / Perplexity / OpenAI
    - all_llama_local:
        reviewer 名称为: Llama 1 / Llama 2 / Llama 3
    - all_qwen_local / qwen_reviewers_gpt5_ac:
        reviewer 名称为: Qwen 1 / Qwen 2 / Qwen 3
    """
    local_models = local_models or {
        "llama": os.getenv("OLLAMA_LLAMA_TAG", "llama3.1:8b"),
        "qwen":  os.getenv("OLLAMA_QWEN_TAG",  "qwen2.5:7b-instruct"),
    }
    local_options = local_options or {"temperature": 0.7, "top_p": 0.9, "num_predict": 1024}
    ollama_host = ollama_host or OLLAMA_HOST

    # 本地 reviewer / 本地 AC
    def _local_llama_reviewer():
        return LLMWrapper(
            "ollama",
            INDEPENDENT_REVIEWER_AGENT,
            512,
            amp_generator_root,
            local_ollama_model=local_models["llama"],
            ollama_host=ollama_host,
            local_options=local_options,
        )

    def _local_qwen_reviewer():
        return LLMWrapper(
            "ollama",
            INDEPENDENT_REVIEWER_AGENT,
            512,
            amp_generator_root,
            local_ollama_model=local_models["qwen"],
            ollama_host=ollama_host,
            local_options=local_options,
        )

    def _ac_local_llama():
        return LLMWrapper(
            "ollama",
            AREA_CHAIR_AGENT,
            2048,
            amp_generator_root,
            local_ollama_model=local_models["llama"],
            ollama_host=ollama_host,
            local_options=local_options,
        )

    def _ac_local_qwen():
        return LLMWrapper(
            "ollama",
            AREA_CHAIR_AGENT,
            2048,
            amp_generator_root,
            local_ollama_model=local_models["qwen"],
            ollama_host=ollama_host,
            local_options=local_options,
        )

    # 远端 reviewer / 远端 AC
    def _api_openai_reviewer():
        return LLMWrapper("openai", INDEPENDENT_REVIEWER_AGENT, 512, amp_generator_root)

    def _api_genai_reviewer():
        return LLMWrapper("genai", INDEPENDENT_REVIEWER_AGENT, 512, amp_generator_root)

    def _api_pplx_reviewer():
        return LLMWrapper("perplexity", INDEPENDENT_REVIEWER_AGENT, 512, amp_generator_root)

    def _ac_openai():
        return LLMWrapper("openai", AREA_CHAIR_AGENT, 2048, amp_generator_root)

    experiment = (experiment or "baseline_api").lower()

    if experiment == "baseline_api":
        # 1) 保留现在的（纯 API）
        agents = OrderedDict({
            "GenAI":      ExpertAgent("GenAI",      _api_genai_reviewer()),
            "Perplexity": ExpertAgent("Perplexity", _api_pplx_reviewer()),
            "OpenAI":     ExpertAgent("OpenAI",     _api_openai_reviewer()),
        })
        meta_reviewer = MetaReviewAgent(_ac_openai())

    elif experiment == "all_llama_local":
        # 2) 全部 Llama（本地）：Llama 1 / 2 / 3
        agents = _make_local_triplet("Llama", _local_llama_reviewer)
        meta_reviewer = MetaReviewAgent(_ac_local_llama())

    elif experiment == "all_qwen_local":
        # 3) 全部 Qwen（本地）：Qwen 1 / 2 / 3
        agents = _make_local_triplet("Qwen", _local_qwen_reviewer)
        meta_reviewer = MetaReviewAgent(_ac_local_qwen())

    elif experiment == "qwen_reviewers_gpt5_ac":
        # 4) 三个本地 Qwen reviewer + OpenAI GPT-5 作为 AC
        agents = _make_local_triplet("Qwen", _local_qwen_reviewer)
        meta_reviewer = MetaReviewAgent(_ac_openai())

    elif experiment == "api_reviewers_qwen_ac":
        # 5) 三个现有 API reviewer + 本地 Qwen AC
        agents = OrderedDict({
            "GenAI":      ExpertAgent("GenAI",      _api_genai_reviewer()),
            "Perplexity": ExpertAgent("Perplexity", _api_pplx_reviewer()),
            "OpenAI":     ExpertAgent("OpenAI",     _api_openai_reviewer()),
        })
        meta_reviewer = MetaReviewAgent(_ac_local_qwen())

    else:
        raise ValueError(f"Unknown experiment mode: {experiment}")

    return agents, meta_reviewer

# =============================================================================
# 主流程（新增 experiment / local_models / ollama_host / local_options）
# =============================================================================
def run_pipeline(
    peptides_info: List[Dict],
    amp_generator_root: str,
    skip_agents: List[str],
    meta: Optional[Dict[str, Any]] = None,
    drift_wordlist: str = '',
    experiment: str = "baseline_api",
    local_models: Optional[Dict[str, str]] = None,
    ollama_host: Optional[str] = None,
    local_options: Optional[Dict[str, Any]] = None
):
    """
    - experiment: 'baseline_api' | 'all_llama_local' | 'all_qwen_local' |
                  'qwen_reviewers_gpt5_ac' | 'api_reviewers_qwen_ac'
    - local_models: {'llama': 'llama3.1:8b', 'qwen': 'qwen2.5:7b-instruct'}
    - ollama_host: 'http://localhost:11434'
    - local_options: {'temperature': 0.7, 'top_p': 0.9, 'num_predict': 1024}
    """
    meta = dict(meta or {})
    if len(skip_agents) == 3:
        return "None", 0.0, {}

    INDEPENDENT_REVIEWER_AGENT, AREA_CHAIR_AGENT = _load_prompt(
        amp_generator_root=amp_generator_root,
        drift=drift_wordlist
    )

    # 构建 Agents
    agents, meta_reviewer = build_agents_by_experiment(
        experiment=experiment,
        INDEPENDENT_REVIEWER_AGENT=INDEPENDENT_REVIEWER_AGENT,
        AREA_CHAIR_AGENT=AREA_CHAIR_AGENT,
        amp_generator_root=amp_generator_root,
        local_models=local_models,
        ollama_host=ollama_host,
        local_options=local_options
    )

    # === 兼容旧的 skip 名称（GenAI/Perplexity/OpenAI） ===
    legacy_order = ["GenAI", "Perplexity", "OpenAI"]
    agent_name_list = list(agents.keys())
    if set(legacy_order).issubset(set(agent_name_list)):
        # 当前实验仍是老三家命名，不需要映射
        normalized_skips = set(skip_agents or [])
    else:
        # 当前实验是 local 命名（Llama i / Qwen i），把旧名按顺序映射为第 1/2/3 个 reviewer
        map_legacy = {
            legacy_order[i]: agent_name_list[i]
            for i in range(min(3, len(agent_name_list)))
        }
        normalized_skips = set(map_legacy.get(s, s) for s in (skip_agents or []))

    _, json_str, _ = build_peptide_json(peptides_info)

    # Area Chair 标签（仅用于日志/计费）
    ac_label_map = {
        "all_llama_local": "Llama AC",
        "all_qwen_local": "Qwen AC",
        "qwen_reviewers_gpt5_ac": "OpenAI AC",
        "api_reviewers_qwen_ac": "Qwen AC",
        "baseline_api": "MetaReviewer",
    }
    ac_label = ac_label_map.get((experiment or "").lower(), "MetaReviewer")

    # 三个 Reviewer 独立评审
    evaluations = {}
    for name, agent in agents.items():
        info = {**meta, "agent": name}
        if name in normalized_skips:
            print(f"\n🧠 {name} skip 评估...")
            evaluations[name] = _load_default_empty_evaluation()
        else:
            print(f"\n🧠 {name} 正在评估...")
            evaluations[name] = agent.generate_response(json_str, meta=info)
            print(f"[{name}]：\n{evaluations[name]}")

    # Area Chair 汇总与打分
    max_retry = 3
    info = {**meta, "agent": ac_label}
    for attempt in range(max_retry):
        meta_review_output = meta_reviewer.synthesize_and_score(evaluations, meta=info)
        print("$" * 80)
        print(ac_label)
        print(meta_review_output)

        meta_review_output = format_area_meta_with_scores(evaluations, meta_review_output)
        lines = meta_review_output.strip().split("\n")
        critical_comments = "".join(lines[:-1])

        try:
            m = re.search(r"(?mi)^\[MetaScore:\s*([+-]?\d+(?:\.\d+)?)\]\s*$", lines[-1])
            critical_score = float(m.group(1))
            break
        except Exception as e:
            print(f"⚠️ 第 {attempt+1} 次解析失败：{e}")
            critical_score = None
            if attempt == max_retry - 1:
                critical_score = 0.0

    print("\n📌 综合评价与评分输出：")
    print(critical_comments)
    print(critical_score)
    return critical_comments, critical_score, evaluations

# =============================================================================
# 示例入口（按需启用）
# =============================================================================
if __name__ == "__main__":
    try:
        from evaluation_service import Evaluator
        evaluator = Evaluator(True)
        all_decoded = [
            "MLKIIRTHMLYWQLRPLARA",
            "DFIIHKDGCDQ",
            "MNALFHLKLFTYKGLIKSFK",
            "MGCGCGSDCNCDGGAQSCSCGDKCDCKGCG",
            "MIKYLGSLLL",
            "MSEVLCSREETCNATHRLKTQTKIYD",
            "MWKTLHQLAAPPRLYQICGRLVPWLAAA",
            "TSASLKFTVLNPKGRIWTMVAGGGASVIYADT",
            "MRNHDLVSDGFLALTAGGLVLLCSSLVALAFG",
            "MNTLKTLVALSFITLFMAVLSAKRSRQAHRQ",
        ]
        evaluation_result, phys_props, tox, amp_act, struct_pred, sim_ana, mic_score, mic_original = \
            evaluator.get_evaluation_outputs_from_agent(all_decoded)
        peptides_info = evaluator.parse_batch_evaluation(evaluation_result)
    except Exception:
        # 无 Evaluator 时的兜底示例
        peptides_info = [
            {
                "sequence": "KKVVKKVVKK",
                "amp_score": 0.7,
                "toxicity_score": 0.1,
                "plddt_score": 0.8,
                "similarity_score": 0.6,
                "mic_original": 12.0,
                "physicochemical_propertie": {"gravy": 0.1, "instability_index": 15.0},
            }
        ]

    # ========== 使用示例（任选其一，取消注释即可） ==========
    # 1) 保持现状（默认）
    # run_pipeline(peptides_info, amp_generator_root=".", skip_agents=[], experiment="baseline_api")

    # 2) 全部 Llama（本地 Ollama）
    # run_pipeline(
    #     peptides_info,
    #     amp_generator_root=".",
    #     skip_agents=[],
    #     experiment="all_llama_local",
    #     local_models={"llama": "llama3.1:8b", "qwen": "qwen2.5:7b-instruct"},
    #     local_options={"temperature": 0.7, "top_p": 0.9, "num_predict": 1200},
    # )

    # 3) 全部 Qwen（本地 Ollama）
    # run_pipeline(
    #     peptides_info,
    #     amp_generator_root=".",
    #     skip_agents=[],
    #     experiment="all_qwen_local",
    #     local_models={"llama": "llama3.1:8b", "qwen": "qwen2.5:7b-instruct"},
    # )

    # 4) 三个本地 Qwen reviewer + GPT-5 AC
    # run_pipeline(
    #     peptides_info,
    #     amp_generator_root=".",
    #     skip_agents=[],
    #     experiment="qwen_reviewers_gpt5_ac",
    #     local_models={"llama": "llama3.1:8b", "qwen": "qwen2.5:7b-instruct"},
    # )

    # 5) 三个 API reviewer + 本地 Qwen AC
    # run_pipeline(
    #     peptides_info,
    #     amp_generator_root=".",
    #     skip_agents=[],
    #     experiment="api_reviewers_qwen_ac",
    #     local_models={"llama": "llama3.1:8b", "qwen": "qwen2.5:7b-instruct"},
    # )
