# cost_tracker.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
import time, json, os, threading, subprocess

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

# ---------- 数据结构 ----------

@dataclass
class LLMCallStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

@dataclass
class TrainEpochStats:
    name: str
    stage: Optional[str]
    iter_idx: Optional[int]
    epoch: Optional[int]
    response_id: Optional[int]
    duration_sec: float
    gpu_count: int
    gpu_hours: float
    peak_mem_mb: Optional[int]
    generated_tokens: int
    samples: int

@dataclass
class ToolRunStats:
    name: str
    stage: Optional[str]
    iter_idx: Optional[int]
    epoch: Optional[int]
    response_id: Optional[int]
    duration_sec: float
    peak_mem_mb: Optional[int]

# ---------- 价格（可选） ----------
# 你可以在运行时设置环境变量 MODEL_PRICING_JSON 指向一个 JSON 文件，覆盖默认价格。
# 价格单位：$/token（比如 0.000005 表示 $5 / 1M tokens）
_DEFAULT_PRICING = {
    # provider/model: {"in": price_per_input_token, "out": price_per_output_token}
    ("openai", "gpt-4o"): {"in": 0.0, "out": 0.0},
    ("openai", "gpt-5"):  {"in": 0.0, "out": 0.0},
    ("perplexity", "sonar"): {"in": 0.0, "out": 0.0},
    ("google", "gemini-2.5-pro"): {"in": 0.0, "out": 0.0},
}

def _load_pricing() -> Dict[Tuple[str, str], Dict[str, float]]:
    path = os.environ.get("MODEL_PRICING_JSON", "").strip()
    table = dict(_DEFAULT_PRICING)
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                if isinstance(k, str) and isinstance(v, dict):
                    prov, mdl = k.split("/", 1) if "/" in k else (k, "")
                    table[(prov.lower(), mdl)] = {"in": float(v.get("in", 0.0)), "out": float(v.get("out", 0.0))}
        except Exception:
            pass
    return table

_PRICING = _load_pricing()

def _estimate_cost(provider: str, model: str, in_tok: int, out_tok: int) -> float:
    key = (provider.lower(), model)
    p = _PRICING.get(key)
    if not p:
        return 0.0
    return float(in_tok) * float(p.get("in", 0.0)) + float(out_tok) * float(p.get("out", 0.0))


# ---------- 全局统计器 ----------
class CostTracker:
    _GLOBAL: "CostTracker" = None

    def __init__(self) -> None:
        # LLM：按 (epoch, stage, agent, api, model) 聚合
        self.llm_calls: Dict[Tuple[Optional[int], Optional[str], str, str, str], LLMCallStats] = defaultdict(LLMCallStats)
        # 每个 PPO epoch 的统计
        self.train_epochs: List[TrainEpochStats] = []
        # 外部工具 / 子进程阶段
        self.tool_runs: List[ToolRunStats] = []
        self.created_at = time.time()

    @classmethod
    def get_global(cls) -> "CostTracker":
        if cls._GLOBAL is None:
            cls._GLOBAL = cls()
        return cls._GLOBAL

    # ---- LLM 记账 ----
    def log_llm_call(
        self,
        *,
        agent: str,
        api: str,           # e.g. "openai.chat.completions", "perplexity.chat.completions", "gemini.generate_content"
        model: str,
        provider: str,      # "openai" / "perplexity" / "google"
        stage: Optional[str] = None,
        epoch: Optional[int] = None,
        iter_idx: Optional[int] = None,
        response_id: Optional[int] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        key = (epoch, stage, agent, api, model)
        s = self.llm_calls[key]
        s.calls += 1
        s.input_tokens += int(input_tokens or 0)
        s.output_tokens += int(output_tokens or 0)
        s.total_tokens += int(input_tokens or 0) + int(output_tokens or 0)
        s.cost_usd += _estimate_cost(provider, model, int(input_tokens or 0), int(output_tokens or 0))

    # ---- 训练 epoch 记账 ----
    def log_train_epoch(
        self,
        *,
        name: str,                   # "ppo_epoch"
        stage: Optional[str],
        iter_idx: Optional[int],
        epoch: Optional[int],
        response_id: Optional[int],
        duration_sec: float,
        gpu_count: int = 1,
        peak_mem_mb: Optional[int] = None,
        generated_tokens: int = 0,
        samples: int = 0,
    ) -> None:
        gpu_hours = gpu_count * duration_sec / 3600.0
        self.train_epochs.append(
            TrainEpochStats(
                name=name,
                stage=stage,
                iter_idx=iter_idx,
                epoch=epoch,
                response_id=response_id,
                duration_sec=duration_sec,
                gpu_count=gpu_count,
                gpu_hours=gpu_hours,
                peak_mem_mb=peak_mem_mb,
                generated_tokens=int(generated_tokens or 0),
                samples=int(samples or 0),
            )
        )

    # ---- 外部工具阶段（可选） ----
    def log_tool_run(
        self,
        *,
        name: str,
        stage: Optional[str],
        iter_idx: Optional[int],
        epoch: Optional[int],
        response_id: Optional[int],
        duration_sec: float,
        peak_mem_mb: Optional[int],
    ) -> None:
        self.tool_runs.append(
            ToolRunStats(
                name=name,
                stage=stage,
                iter_idx=iter_idx,
                epoch=epoch,
                response_id=response_id,
                duration_sec=duration_sec,
                peak_mem_mb=peak_mem_mb,
            )
        )

    # ---- 导出 ----
    def dump_json(self, path: str = "compute_costs.json") -> None:
        data = {
            "created_at": self.created_at,
            "llm_calls": [
                {
                    "epoch": k[0], "stage": k[1], "agent": k[2], "api": k[3], "model": k[4],
                    "calls": v.calls, "input_tokens": v.input_tokens, "output_tokens": v.output_tokens,
                    "total_tokens": v.total_tokens, "cost_usd": round(v.cost_usd, 6),
                } for k, v in self.llm_calls.items()
            ],
            "train_epochs": [asdict(x) for x in self.train_epochs],
            "tool_runs": [asdict(x) for x in self.tool_runs],
            "total_gpu_hours": sum(x.gpu_hours for x in self.train_epochs),
            "total_llm_cost_usd": round(sum(v.cost_usd for v in self.llm_calls.values()), 6),
            "total_llm_tokens": sum(v.total_tokens for v in self.llm_calls.values()),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# ---------- GPU 峰值显存采样（nvidia-smi） ----------
def _query_max_gpu_mem_mb_once() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, encoding="utf-8"
        )
    except Exception:
        return None
    vals = []
    for ln in out.strip().splitlines():
        try:
            vals.append(int(ln.strip()))
        except Exception:
            pass
    return max(vals) if vals else None

class _NVSMIPeakSampler:
    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    def start(self):
        def _run():
            while not self._stop.is_set():
                v = _query_max_gpu_mem_mb_once()
                if v is not None and v > self.peak:
                    self.peak = v
                time.sleep(self.interval)
        self._thr = threading.Thread(target=_run, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=0.2)

class gpu_phase:
    """
    用于外部工具/子进程的阶段计时 + GPU 峰值显存采样（基于 nvidia-smi 轮询）
    """
    def __init__(self, name: str, *, stage: Optional[str] = None,
                 iter_idx: Optional[int] = None, epoch: Optional[int] = None,
                 response_id: Optional[int] = None, tracker: Optional[CostTracker] = None,
                 interval: float = 1.0):
        self.name, self.stage, self.iter_idx, self.epoch, self.response_id = name, stage, iter_idx, epoch, response_id
        self.tracker = tracker or CostTracker.get_global()
        self.interval = interval
        self.t0 = None
        self.sampler = _NVSMIPeakSampler(interval=self.interval)

    def __enter__(self):
        self.t0 = time.time()
        self.sampler.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.sampler.stop()
        dt = time.time() - self.t0 if self.t0 else 0.0
        self.tracker.log_tool_run(
            name=self.name, stage=self.stage, iter_idx=self.iter_idx,
            epoch=self.epoch, response_id=self.response_id,
            duration_sec=dt, peak_mem_mb=(self.sampler.peak or None),
        )
        return False

class cuda_peak_meter:
    """
    用于单进程内（PPO训练）统计：torch 的峰值显存 + 时长
    """
    def __init__(self):
        self.t0 = None
        self.peak_mb = None

    def __enter__(self):
        self.t0 = time.time()
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                peak = torch.cuda.max_memory_allocated()
                self.peak_mb = int(peak / (1024 * 1024))
            except Exception:
                self.peak_mb = None
        return False

    @property
    def duration(self) -> float:
        return (time.time() - self.t0) if self.t0 else 0.0

# ---------- LLM 返回的 usage 统一解析 ----------
def log_openai_completion(resp, *, agent: str, stage: Optional[str], epoch: Optional[int],
                          iter_idx: Optional[int], response_id: Optional[int], api: str = "openai.chat.completions") -> None:
    usage = getattr(resp, "usage", None)
    in_tok = out_tok = 0
    model = getattr(resp, "model", "")
    if usage is not None:
        in_tok = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    CostTracker.get_global().log_llm_call(
        agent=agent, api=api, model=model or "", provider="openai",
        stage=stage, epoch=epoch, iter_idx=iter_idx, response_id=response_id,
        input_tokens=int(in_tok), output_tokens=int(out_tok),
    )

def log_perplexity_completion(response_json: Dict[str, Any], *, agent: str, stage: Optional[str],
                              epoch: Optional[int], iter_idx: Optional[int], response_id: Optional[int]) -> None:
    usage = (response_json or {}).get("usage", {}) if isinstance(response_json, dict) else {}
    in_tok = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
    out_tok = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
    model = (response_json or {}).get("model", "sonar")
    CostTracker.get_global().log_llm_call(
        agent=agent, api="perplexity.chat.completions", model=str(model), provider="perplexity",
        stage=stage, epoch=epoch, iter_idx=iter_idx, response_id=response_id,
        input_tokens=int(in_tok), output_tokens=int(out_tok),
    )

def log_gemini_response(resp, *, agent: str, stage: Optional[str], epoch: Optional[int],
                        iter_idx: Optional[int], response_id: Optional[int]) -> None:
    # Google Generative AI: usage_metadata.prompt_token_count / candidates_token_count
    in_tok = out_tok = 0
    try:
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            in_tok = int(getattr(um, "prompt_token_count", 0) or 0)
            out_tok = int(getattr(um, "candidates_token_count", 0) or 0)
    except Exception:
        pass
    CostTracker.get_global().log_llm_call(
        agent=agent, api="gemini.generate_content", model="gemini-2.5-pro", provider="google",
        stage=stage, epoch=epoch, iter_idx=iter_idx, response_id=response_id,
        input_tokens=int(in_tok), output_tokens=int(out_tok),
    )
