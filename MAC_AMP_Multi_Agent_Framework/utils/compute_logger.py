from __future__ import annotations
"""
compute_logger.py

一个简单的“全局开销记录器”，用于统计：
- GPU 训练时间（近似 GPU hours）
- 训练过程的峰值显存（通过 nvidia-smi 轮询）
- 每个 epoch / stage / agent / API 的调用次数 & token 数量

使用方式建议：
1）在项目某处（例如 utils/compute_logger.py）保存本文件
2）在 main_workflow.py / rl_scientist_reward_designer.py 等文件中：
    from utils.compute_logger import compute_logger
3）在合适位置调用：
    - compute_logger.set_context(epoch=..., stage=..., agent=...)
    - compute_logger.start_gpu_block(...); 运行训练脚本; compute_logger.end_gpu_block()
    - 在每次 LLM API 返回后调用 compute_logger.log_openai_response(...)
4）在程序结束时：
    compute_logger.save_json("compute_stats.json")
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from collections import defaultdict
import json
import time
import threading
import subprocess
import os


# ---------- 数据结构 ----------

@dataclass
class APICallStats:
    count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, in_tok: int = 0, out_tok: int = 0) -> None:
        self.count += 1
        self.input_tokens += int(in_tok or 0)
        self.output_tokens += int(out_tok or 0)


@dataclass
class GPUBlock:
    tag: str
    epoch: Optional[int]
    stage: Optional[str]
    gpu_index: Optional[int]
    gpu_count: int = 1
    start_time: float = 0.0
    end_time: float = 0.0
    duration_sec: float = 0.0
    peak_mem_mb: int = 0


class ComputeLogger:
    """
    负责全局收集各种开销信息。
    - 线程安全需求不高，本项目基本单线程（除了内部监控显存的线程）。
    """
    def __init__(self) -> None:
        # 当前上下文（可选）
        self.current_epoch: Optional[int] = None
        self.current_stage: Optional[str] = None
        self.current_agent: Optional[str] = None

        # 嵌套字典: epoch -> stage -> agent -> api_name -> APICallStats
        self._api_stats = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(APICallStats)
                )
            )
        )

        # OpenAI 调用总览
        self.total_api_calls: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

        # GPU block 记录
        self.gpu_blocks: List[GPUBlock] = []
        self._current_gpu_block: Optional[GPUBlock] = None
        self._gpu_watch_thread: Optional[threading.Thread] = None
        self._gpu_watch_stop = threading.Event()
        self.global_peak_mem_mb: int = 0
        self.total_gpu_seconds: float = 0.0

        # 实验起始时间
        self.start_time: float = time.time()

    # ----- 上下文管理 -----
    def set_context(
        self,
        *,
        epoch: Optional[int] = None,
        stage: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> None:
        """
        设置“当前上下文”，后续未显式传入 epoch/stage/agent 的 log 调用会用到。
        """
        if epoch is not None:
            self.current_epoch = epoch
        if stage is not None:
            self.current_stage = stage
        if agent is not None:
            self.current_agent = agent

    class _Context:
        def __init__(self, logger: "ComputeLogger", epoch, stage, agent) -> None:
            self.logger = logger
            self.new = dict(epoch=epoch, stage=stage, agent=agent)
            self.old = dict(
                epoch=logger.current_epoch,
                stage=logger.current_stage,
                agent=logger.current_agent,
            )

        def __enter__(self):
            self.logger.set_context(**self.new)
            return self.logger

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.logger.set_context(**self.old)

    def context(
        self,
        *,
        epoch: Optional[int] = None,
        stage: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> "_Context":
        """
        语法糖：
        with compute_logger.context(epoch=1, stage="sandbox_1", agent="RLScientist"):
            ... 调用若干 API ...
        """
        return self._Context(self, epoch, stage, agent)

    # ----- OpenAI API 调用日志 -----
    def log_api_call(
        self,
        *,
        api_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        epoch: Optional[int] = None,
        stage: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> None:
        """记录一次 API 调用的统计信息。"""
        e = self.current_epoch if epoch is None else epoch
        s = self.current_stage if stage is None else stage
        a = self.current_agent if agent is None else agent

        # 为了 JSON 友好，None 用字符串 "None" 替代
        e_key = "None" if e is None else int(e)
        s_key = "None" if s is None else str(s)
        a_key = "None" if a is None else str(a)
        api_key = str(api_name)

        stats: APICallStats = self._api_stats[e_key][s_key][a_key][api_key]
        stats.add(input_tokens, output_tokens)

        self.total_api_calls += 1
        self.total_input_tokens += int(input_tokens or 0)
        self.total_output_tokens += int(output_tokens or 0)

    def log_openai_response(
        self,
        resp: Any,
        *,
        api_name: Optional[str] = None,
        epoch: Optional[int] = None,
        stage: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> None:
        """
        针对 OpenAI ChatCompletion 返回值的便捷方法：
        自动从 resp.usage 中提取 input/output tokens 并调用 log_api_call。
        """
        usage = getattr(resp, "usage", None)
        in_tok = out_tok = 0
        if usage is not None:
            # 新版 SDK: input_tokens / output_tokens；旧版: prompt_tokens / completion_tokens
            in_tok = getattr(usage, "input_tokens", None)
            if in_tok is None:
                in_tok = getattr(usage, "prompt_tokens", 0)
            out_tok = getattr(usage, "output_tokens", None)
            if out_tok is None:
                out_tok = getattr(usage, "completion_tokens", 0)

        self.log_api_call(
            api_name=api_name or getattr(resp, "model", "unknown_model"),
            input_tokens=int(in_tok or 0),
            output_tokens=int(out_tok or 0),
            epoch=epoch,
            stage=stage,
            agent=agent,
        )

    # ----- GPU 训练 block 记录 -----
    def start_gpu_block(
        self,
        *,
        tag: str,
        epoch: Optional[int] = None,
        stage: Optional[str] = None,
        gpu_index: Optional[int] = None,
        gpu_count: int = 1,
        poll_interval_sec: float = 1.0,
    ) -> None:
        """
        开始记录一个“GPU 训练区间”（例如一次 run_ppo_training 调用）。
        - tag: 任意标记（例如 reward_id / 'warmup_0'）
        - epoch / stage: 方便后续汇总
        - gpu_index: 要监控的 GPU 编号；若为 None，则从 CUDA_VISIBLE_DEVICES 推断
        """
        if self._current_gpu_block is not None:
            # 简单保护：避免嵌套或重复 start
            return

        if gpu_index is None:
            # 例如 CUDA_VISIBLE_DEVICES="2,3" -> 2
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
            try:
                gpu_index = int(visible)
            except Exception:
                gpu_index = 0

        block = GPUBlock(
            tag=str(tag),
            epoch=epoch,
            stage=stage,
            gpu_index=gpu_index,
            gpu_count=int(gpu_count) if gpu_count is not None else 1,
            start_time=time.time(),
        )
        self._current_gpu_block = block
        self._gpu_watch_stop.clear()

        def _watch():
            # 轮询 nvidia-smi，粗略估计峰值显存
            while not self._gpu_watch_stop.is_set():
                try:
                    out = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "-i",
                            str(block.gpu_index),
                            "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        encoding="utf-8",
                    )
                    first_line = out.strip().splitlines()[0]
                    mem_mb = int(first_line)
                    if mem_mb > block.peak_mem_mb:
                        block.peak_mem_mb = mem_mb
                    if mem_mb > self.global_peak_mem_mb:
                        self.global_peak_mem_mb = mem_mb
                except Exception:
                    # 没有 nvidia-smi 或其它错误时，忽略
                    pass
                self._gpu_watch_stop.wait(poll_interval_sec)

        self._gpu_watch_thread = threading.Thread(target=_watch, daemon=True)
        self._gpu_watch_thread.start()

    def end_gpu_block(self) -> None:
        """结束当前 GPU 区间，累积 GPU seconds 与显存信息。"""
        block = self._current_gpu_block
        if block is None:
            return

        block.end_time = time.time()
        block.duration_sec = max(0.0, block.end_time - block.start_time)
        self.total_gpu_seconds += block.duration_sec * max(1, block.gpu_count)

        self._gpu_watch_stop.set()
        if self._gpu_watch_thread is not None:
            try:
                self._gpu_watch_thread.join(timeout=2.0)
            except Exception:
                pass

        self.gpu_blocks.append(block)
        self._current_gpu_block = None
        self._gpu_watch_thread = None

    # ----- 导出 / 汇总 -----
    @property
    def total_gpu_hours(self) -> float:
        return self.total_gpu_seconds / 3600.0

    def api_stats_dict(self) -> Dict[str, Any]:
        """
        把嵌套 defaultdict 转成普通 dict，方便 JSON dump。
        """
        out: Dict[str, Any] = {}
        for epoch, d_stage in self._api_stats.items():
            out_epoch: Dict[str, Any] = {}
            for stage, d_agent in d_stage.items():
                out_stage: Dict[str, Any] = {}
                for agent, d_api in d_agent.items():
                    out_agent: Dict[str, Any] = {}
                    for api, stats in d_api.items():
                        out_agent[api] = asdict(stats)
                    out_stage[agent] = out_agent
                out_epoch[stage] = out_stage
            out[str(epoch)] = out_epoch
        return out

    def gpu_blocks_dict(self) -> List[Dict[str, Any]]:
        return [asdict(b) for b in self.gpu_blocks]

    def summary(self) -> Dict[str, Any]:
        return {
            "experiment_start_time": self.start_time,
            "total_api_calls": self.total_api_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "api_stats": self.api_stats_dict(),
            "total_gpu_seconds": self.total_gpu_seconds,
            "total_gpu_hours": self.total_gpu_hours,
            "global_peak_mem_mb": self.global_peak_mem_mb,
            "gpu_blocks": self.gpu_blocks_dict(),
        }

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, ensure_ascii=False, indent=2)


# 提供一个全局实例，代码里直接 from utils.compute_logger import compute_logger 即可使用
compute_logger = ComputeLogger()
