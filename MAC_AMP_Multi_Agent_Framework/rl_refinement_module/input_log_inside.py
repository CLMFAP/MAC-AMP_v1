from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Literal
import math
import copy
import re

# ---------- 小工具 ----------
def _fmt(x):
    """简洁数值格式，避免 0.5780000000000001。"""
    return f"{x:.6g}" if isinstance(x, float) else str(x)

def _one_line_code(code: str) -> str:
    """把源码压成单行，去掉多余空白，便于嵌入方括号中展示。"""
    s = code.strip().replace("```", "")
    s = re.sub(r"\s+", " ", s)
    return s

# ---------- 两种 epoch 结构 ----------
@dataclass
class EpochScientist:
    epoch: int
    mic_raw: float             # Raw MIC Value
    mic_score: float           # MIC Score（映射后的得分）
    amp_score: float           # AMP Score
    meta_review_score: float   # Meta Review Score
    stage: Optional[str] = None  # 新增：每个 epoch 的 Stage（可为 "0","1","2","i","3",...）

@dataclass
class EpochCritic:
    epoch: int
    review_message: str        # 仅文本消息（无 score）
    stage: Optional[str] = None  # 新增：每个 epoch 的 Stage


# ---------- 每个 run 同时容纳两类 epoch，互不混 ----------
@dataclass
class RunRecord:
    reward_id: str
    code_str: str
    meta: Dict = field(default_factory=dict)
    sci_epochs: List[EpochScientist] = field(default_factory=list)
    critic_epochs: List[EpochCritic] = field(default_factory=list)


class InputLog:
    """
    一个“谱系容器”：内部可维护多套 reward 记录（runs）：
      - 每套记录包含：reward_id、code_str、scientist/critic 两类 epoch
      - fork_as_independent: 基于 parent 的历史复制为“独立容器”，新容器里可保留父 run 作为对照
    必要方法：add_run, add_epoch_scientist, add_epoch_critic, fork_as_independent,
            to_str_scientist, to_str_critic
    """
    def __init__(self, role: str = "scientist"):
        self.role = role  # "scientist" / "critic"（仅作标注，不影响格式）
        self._runs: List[RunRecord] = []

    # 新增一套 reward function
    def add_run(self, reward_id: str, code_str: str, meta: Optional[Dict] = None) -> None:
        if any(r.reward_id == reward_id for r in self._runs):
            raise ValueError(f"reward_id '{reward_id}' already exists.")
        self._runs.append(RunRecord(reward_id=reward_id, code_str=code_str, meta=meta or {}))

    # 追加 scientist 的 epoch（固定四字段）
    def add_epoch_scientist(self,
                        reward_id: str, epoch: int,
                        mic_raw: float, mic_score: float,
                        amp_score: float, meta_review_score: float,
                        stage: Optional[str] = None) -> None:
        run = next((r for r in self._runs if r.reward_id == reward_id), None)
        if run is None:
            raise KeyError(f"reward_id '{reward_id}' not found. Call add_run(...) first.")
        run.sci_epochs.append(EpochScientist(epoch, mic_raw, mic_score, amp_score, meta_review_score, stage))

    # 追加 critic 的 epoch（仅文本消息）
    def add_epoch_critic(self, reward_id: str, epoch: int, review_message: str,
                     stage: Optional[str] = None) -> None:
        run = next((r for r in self._runs if r.reward_id == reward_id), None)
        if run is None:
            raise KeyError(f"reward_id '{reward_id}' not found. Call add_run(...) first.")
        run.critic_epochs.append(EpochCritic(epoch, review_message.strip(), stage))

    # 基于父 run 生成“独立容器”（默认不复制父的 epoch；可选择是否包含父 run 作为对照）
    def fork_as_independent(self,
                            parent_reward_id: str,
                            new_reward_id: str,
                            new_code_str: Optional[str] = None,
                            copy_epochs: bool = False,
                            include_parent: bool = False,
                            extra_meta: Optional[Dict] = None) -> "InputLog":
        """
        以 parent 为蓝本，创建一个“独立的 InputLog”（新的对象）。
          - include_parent=True: 会把父 run 深拷贝进新容器（用于展示对照）
          - copy_epochs=True : 拷贝父的历史 epoch；False: 新 run 从零开始
        """
        parent = next((r for r in self._runs if r.reward_id == parent_reward_id), None)
        if parent is None:
            raise KeyError(f"parent_reward_id '{parent_reward_id}' not exists.")

        # 新容器
        new_log = InputLog(role=self.role)

        # 可选：把父 run 整体拷贝到新容器中（仅作对照，不会与新 run 混）
        if include_parent:
            new_log._runs.append(copy.deepcopy(parent))

        # 新 run
        code = new_code_str if new_code_str is not None else parent.code_str
        meta = {**parent.meta, "parent": parent.reward_id, **(extra_meta or {})}
        if copy_epochs:
            sci = copy.deepcopy(parent.sci_epochs)
            cri = copy.deepcopy(parent.critic_epochs)
        else:
            sci, cri = [], []
        new_log._runs.append(RunRecord(new_reward_id, code, meta, sci, cri))
        return new_log

    # ===== 固定格式：scientist（单行；含 reward_id；Code 后带空格）=====
    def to_str_scientist(self, reward_id: Optional[str] = None) -> str:
        """
        逐行：
        [Stage: {stage}][Reward Function:{reward_id}][Epoch: {k}]
        [Raw MIC Value: {..}, MIC Score: {..}][AMP Score: {..}][Meta Review Score: {..}]
        ... [/]
        """
        
        runs = self._runs if reward_id is None else [r for r in self._runs if r.reward_id == reward_id]
        if not runs:
            return "[]"
        lines: List[str] = []
        for r in runs:
            parts = []
            # stage_str = f"[Stage: {e.stage}]" if getattr(e, "stage", None) is not None else ""
            stage_str = (lambda _ep=next(iter(r.sci_epochs), None): f"[Stage: {_ep.stage}]" if getattr(_ep, "stage", None) is not None else "")()
            parts.append(f"{stage_str}[Reward Function:]{_one_line_code(r.code_str)}\n")
            for e in sorted(r.sci_epochs, key=lambda x: x.epoch):
                parts.append(
                    # f"[Epoch: {e.epoch}]"
                    f"[Raw MIC Value: {_fmt(e.mic_raw)}, MIC Score: {_fmt(e.mic_score)}]\n"
                    f"[AMP Score: {_fmt(e.amp_score)}]\n"
                    f"[Meta Review Score: {_fmt(e.meta_review_score)}]\n"
                )
                parts.append("[/]\n")
            lines.append(" ".join(p for p in parts if p))
            # [M3][Stage: {IN}][Reward Function:]{F3}
            # [Raw MIC Value: 8.6, MIC Score: 0.104]
            # [AMP Score: 0.540]
            # [Meta Review Score: -0.012]
            # [/]
        return "\n".join(lines)

    # ===== 固定格式：critic（单行；Code 后不留空格，按你要求）=====
    def to_str_critic(self, reward_id: Optional[str] = None) -> str:
        """
        逐行：
        [Stage: {stage}][Reward Function:{reward_id}][Epoch: {k}][Meta Review Message:{...}]
        ... [/]
        """
        runs = self._runs if reward_id is None else [r for r in self._runs if r.reward_id == reward_id]
        if not runs:
            return "[]"
        lines: List[str] = []
        for r in runs:
            parts = []
            for e in sorted(r.critic_epochs, key=lambda x: x.epoch):
                # stage_str = f"[Stage: {e.stage}]" if getattr(e, "stage", None) is not None else ""
                parts.append(
                    # f"{stage_str}[Reward Function:]{_one_line_code(r.code_str)}"
                    # f"[Epoch: {e.epoch}]"
                    f"[Meta Review Message:]\n{e.review_message}]\n"
                )
                parts.append("[/]\n")
            lines.append(" ".join(p for p in parts if p))

        # [Meta Review Message:]
        # [EFF:][meta_comment]三方一致认为AMP概率偏低；净电荷多偏低（R1/R2），R3评为optimal属次要分歧。MIC带宽上R1/R2判为high而R3判为low存在显著不一致；疏水性两位评审均为balanced。样本层面X7/X8较优，但整体效力偏弱。[EFF_Tags: (core=mic_band, states=(high|high|low)) | (core=amp_likelihood, states=(low|low|low)) | (core=net_charge, states=(low|low|optimal)) | (core=hydrophobicity, states=(balanced|balanced))][MetaScore_EFF: -0.332]
        # [Safe:][meta_comment]对过度阳离子化均判为低风险，芳香性富集两方为低；毒性上R1/R2多为medium而R3偏low，且均指出X2/X4风险较高；疏水性安全R1评high而R3为balanced有差异。整体安全性中等偏好但需关注X2/X4。[Safe_Tags: (core=toxinpred, states=(medium|medium|low)) | (core=hydrophobicity_safety, states=(high|balanced)) | (core=over_cationic_level, states=(low|low|low)) | (core=aromatic_enrichment, states=(low|low))][MetaScore_Safe: 0.302]
        # [DevStruct:][meta_comment]pLDDT多为中-高（R1/R2中等，R3较高）；长度R2/R3一致为optimal；不稳定指数R2/R3为medium但R1为high；溶解度R1评poor而R3为moderate；R1单独指出X4半胱氨酸复杂度高。总体可开发性尚可但存溶解与稳定性风险。[DS_Tags: (core=plddt, states=(medium|medium|high)) | (core=instability_index, states=(high|medium|medium)) | (core=solubility_proxy, states=(poor|moderate)) | (core=cysteine_complexity, states=(high)) | (core=length_band, states=(optimal|optimal))][MetaScore_DS: 0.095]
        # [Orig:][meta_comment]三方均认与模板相似度中-高（R1高，R2/R3中）；批内多样性R2/R3为高而R1为中；低复杂度含量R1高、R2低呈对立。X8/X9相似度偏高为共识。整体原创性受模板相似度限制，内部多样性尚可。[Orig_Tags: (core=foldseek_similarity, states=(high|medium|medium)) | (core=batch_diversity, states=(medium|high|high)) | (core=low_complexity_content, states=(high|low)) | (core=kmer_reuse, states=(medium))][MetaScore_Orig: 0.114]
        # [/]
        return "\n".join(lines)
    
    def last_code_str(self, prefer: Literal["append", "epoch"] = "append") -> Optional[str]:
        """
        返回“最后一个 run”的 code_str。
        - prefer="append": 取按 add_run 追加顺序的最后一个 run
        - prefer="epoch" : 取训练进度最新的 run（比较 sci/critic 的最后 epoch）
        """
        if not self._runs:
            return None
        if prefer == "append":
            return self._runs[-1].code_str
        # prefer == "epoch"
        def _last_ep(r: RunRecord) -> int:
            s = max((e.epoch for e in r.sci_epochs), default=0)
            c = max((e.epoch for e in r.critic_epochs), default=0)
            return max(s, c)
        best = max(self._runs, key=_last_ep)
        return best.code_str
    
    def last_reward_id(self) -> Optional[str]:
        """
        返回“最后一个 run”的 code_str。
        - prefer="append": 取按 add_run 追加顺序的最后一个 run
        - prefer="epoch" : 取训练进度最新的 run（比较 sci/critic 的最后 epoch）
        """
        if not self._runs:
            return None
        return self._runs[-1].reward_id

    def select_best_by_response_rid(self,
                                    llm_response: str,
                                    sci_logs: List["InputLog"],
                                    cri_logs: List["InputLog"],
                                    ) -> (str, "InputLog", "InputLog", int):
        """
        从 llm_response 中抽取推荐的 reward_id（新格式：
        iter_{iter}_sandbox_iter_{sandbox_iter}_response_{response_id}），
        在 (sci_logs, cri_logs) 里选择包含该 reward_id 的最佳一对：
          - 至少要 scientist 和 critic 两边都存在对应 run
          - 按 (sci_ep_cnt, cri_ep_cnt, last_sci_ep, last_cri_ep) 降序择优
        返回: (reward_id, best_sci_log, best_cri_log, best_index)
        """

        # 规范化：iter_001_sandbox_iter_07_response_0009 -> iter_1_sandbox_iter_7_response_9
        def _canon_new_rid_local(rid: str) -> str:
            m = re.fullmatch(r'(?i)iter_(\d+)_sandbox_iter_(\d+)_response_(\d+)', rid.strip())
            if not m:
                raise ValueError(f"Unrecognized new-format reward_id: {rid}")
            it, sb, resp = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return f"iter_{it}_sandbox_iter_{sb}_response_{resp}"

        # 宽松规范化：不符合新格式则返回 None（用于比较时跳过）
        def _canon_or_none_local(rid: str):
            try:
                return _canon_new_rid_local(rid)
            except Exception:
                return None

        m = re.search(
            r'(?i)Recommended function reward_id:\s*(iter_\d+_sandbox_iter_\d+_response_\d+)',
            llm_response
        )
        if not m:
            raise ValueError("No new-format reward_id found in llm_response "
                             "(expect like: iter_1_sandbox_iter_7_response_9)")
        rid = _canon_new_rid_local(m.group(1))

        # 在两个列表里挑 run 最全、最后 epoch 最大的那一对
        candidates = []
        for i, (sci, cri) in enumerate(zip(sci_logs, cri_logs)):
            def _find_run(log: "InputLog"):
                r = next((r for r in log._runs if _canon_or_none_local(r.reward_id) == rid), None)
                return r
            rs, rc = _find_run(sci), _find_run(cri)
            if rs is None or rc is None:
                continue
            sci_ep_cnt = len(rs.sci_epochs)
            cri_ep_cnt = len(rc.critic_epochs)
            last_sci_ep = max((e.epoch for e in rs.sci_epochs), default=0)
            last_cri_ep = max((e.epoch for e in rc.critic_epochs), default=0)
            candidates.append((i, sci, cri, sci_ep_cnt, cri_ep_cnt, last_sci_ep, last_cri_ep))

        if not candidates:
            raise ValueError(f"No pair of logs contain reward_id '{rid}' in required format.")

        candidates.sort(key=lambda t: (t[3], t[4], t[5], t[6]), reverse=True)
        best_index, best_sci, best_cri, *_ = candidates[0]
        return rid, best_sci, best_cri, best_index
    
    # 取“最后一个 run 的最后一个 scientist epoch”的 (amp, mic, meta) 三元组
    def _last_metrics_triplet(self) -> Optional[Tuple[float, float, float]]:
        if not self._runs:
            return None
        last_run = self._runs[-1]
        if not last_run.sci_epochs:
            return None
        last_ep = max(last_run.sci_epochs, key=lambda e: e.epoch)
        return (last_ep.amp_score, last_ep.mic_score, last_ep.meta_review_score)

    # 若 self 在三个分量上都严格小于 other（允许一个很小的容差 eps），则返回 True
    def is_strictly_worse_than(self, other: "InputLog", eps: float = 0.0) -> bool:
        a = self._last_metrics_triplet()
        b = other._last_metrics_triplet()
        if a is None or b is None:
            return False  # 无法比较则返回 False（也可改成抛错，看你偏好）
        # 过滤 NaN/inf
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (*a, *b)):
            return False
        # 严格小于（带容差）：a_i + eps < b_i 对全部三项成立
        return all((ai + eps) < bi for ai, bi in zip(a, b))


import os, re

def _canon_new_rid(rid: str) -> str:
    """
    规范化新格式 reward_id：
    'iter_001_sandbox_iter_07_response_0009' -> 'iter_1_sandbox_iter_7_response_9'
    """
    m = re.fullmatch(r'(?i)iter_(\d+)_sandbox_iter_(\d+)_response_(\d+)', rid.strip())
    if not m:
        raise ValueError(f"Unrecognized new-format reward_id: {rid}")
    it, sb, resp = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f"iter_{it}_sandbox_iter_{sb}_response_{resp}"

def load_memory_log_file(filepath: str,
                         sci_log: "InputLog",
                         cri_log: "InputLog",
                         default_code_str: str = "loaded_from_memory_log") -> None:
    """
    读取 txt 日志：iter_{i}_sandbox_iter_{j}_response_{k}_memory_log.txt
    并把内容追加到两个 InputLog 容器：
      - sci_log: role="scientist"
      - cri_log: role="critic"

    规则：
      - reward_id = f"iter_{i}_sandbox_iter_{j}_response_{k}"
      - 仅支持 .txt，每一非空行视作一个 epoch
    """
    # --- 1) 从文件名推断 reward_id（严格新格式 + 仅 .txt） ---
    base = os.path.basename(filepath)
    m = re.fullmatch(r'(iter_\d+_sandbox_iter_\d+_response_\d+)_memory_log\.txt', base)
    if not m:
        raise ValueError(
            f"Filename does not match pattern: {base} ; "
            "expect 'iter_{i}_sandbox_iter_{j}_response_{k}_memory_log.txt'"
        )
    reward_id = _canon_new_rid(m.group(1))

    # --- 2) 确保两个 log 都有该 run ---
    def _get_or_add_run(log: "InputLog", rid: str):
        if not any(r.reward_id == rid for r in log._runs):
            log.add_run(rid, default_code_str)
    _get_or_add_run(sci_log, reward_id)
    _get_or_add_run(cri_log, reward_id)

    # --- 3) 起始 epoch（分别以各自已有条目计数） ---
    def _next_epoch_for(log: "InputLog", rid: str, sci: bool) -> int:
        run = next((r for r in log._runs if r.reward_id == rid), None)
        if run is None:
            return 1
        n = len(run.sci_epochs) if sci else len(run.critic_epochs)
        return n + 1
    sci_epoch = _next_epoch_for(sci_log, reward_id, sci=True)
    cri_epoch = _next_epoch_for(cri_log, reward_id, sci=False)

    # --- 4) 解析一行 ---
    def _parse_line(line: str):
        """
        期望行里包含（尽量多的）字段：
          [Stage: ...][Epoch: ...][Raw MIC Value: ...][MIC Score: ...][AMP Score: ...][Meta Review Score: ...][Meta Review Message: ...]
        其中 Stage 可能是 '0','1','2','3','i' 等字符串；Epoch 通常是整数。
        """
        s = line.strip()
        if not s:
            return None

        def _grab(pattern: str):
            mm = re.search(pattern, s)
            return mm.group(1).strip() if mm else None

        def _grab_float(pattern: str):
            mm = re.search(pattern, s)
            try:
                return float(mm.group(1)) if mm else None
            except Exception:
                return None

        # 解析 Stage（允许任意非右括号的内容）
        stage = _grab(r"\[Stage:\s*([^\]]+)\]")

        # 解析 Epoch：既兼容 [Epoch: 0] 也兼容 [Epoch 0]
        ep_str = _grab(r"\[Epoch[:\s]+(\d+)\]")
        ep = int(ep_str) if ep_str is not None else None

        mic_raw = _grab_float(r"\[Raw MIC Value:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
        mic_score = _grab_float(r"\bMIC Score:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
        amp_score = _grab_float(r"\bAMP Score:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
        meta_review_score = _grab_float(r"\bMeta Review Score:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")

        msg = None
        idx = s.find("[Meta Review Message:")
        if idx != -1:
            msg = s[idx + len("[Meta Review Message:"):].strip()
            msg = msg.rstrip("]").strip()

        nums = [mic_raw, mic_score, amp_score, meta_review_score]
        if msg is None or sum(x is not None for x in nums) < 3:
            return None

        return (
            stage,
            ep,  # 可能为 None，外面用自增回退
            mic_raw if mic_raw is not None else 0.0,
            mic_score if mic_score is not None else 0.0,
            amp_score if amp_score is not None else 0.0,
            meta_review_score if meta_review_score is not None else 0.0,
            msg
        )

    # --- 5) 读取 txt 并逐行追加 ---
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]

    for ln in lines:
        parsed = _parse_line(ln)
        if not parsed:
            continue
        stage, parsed_ep, mic_raw, mic_score, amp_score, meta_review_score, msg = parsed

        # epoch 号优先使用日志行里的 [Epoch: k]；没有就回退到自增
        sci_ep = parsed_ep if parsed_ep is not None else sci_epoch
        cri_ep = parsed_ep if parsed_ep is not None else cri_epoch

        sci_log.add_epoch_scientist(reward_id, sci_ep, mic_raw, mic_score, amp_score, meta_review_score, stage=stage)
        cri_log.add_epoch_critic(reward_id, cri_ep, msg, stage=stage)

        # 仅在未从日志中读取 epoch 时推进自增计数器
        if parsed_ep is None:
            sci_epoch += 1
            cri_epoch += 1

import os, re
from typing import Optional, Literal

def env_source_path_from_last_run(
    log,
    *,
    base_dir: Optional[str] = None,
    prefer: Literal["append", "epoch"] = "append",
) -> str:
    """
    根据 InputLog 中“最后一个 run 的 reward_id”（新统一格式）
    生成源码快照路径：{base_dir}/<reward_id>.py

    新格式: iter_{iter}_sandbox_iter_{sandbox_iter}_response_{response_id}
    例: 
    iter_2_sandbox_iter_2_response_0 -> {base_dir}/iter_2_sandbox_iter_2_response_0.py
    """
    # 找到最后一个 run
    if not log._runs:
        raise ValueError("InputLog has no runs.")
    last = log._runs[-1]
    rid = last.reward_id

    # 规范化 rid（确保新格式）
    m = re.fullmatch(r'(?i)iter_(\d+)_sandbox_iter_(\d+)_response_(\d+)', rid.strip())
    if not m:
        raise ValueError(f"Last reward_id is not in new format: {rid}")

    fname = f"{rid}.py"
    if base_dir is None:
        base_dir = os.getcwd()
    return os.path.join(base_dir, fname)

import re

def pick_best_logs_by_reward_id(llm_response: str,
                                sci_logs: list,
                                cri_logs: list):
    """
    从 llm_response 中抽取推荐的 reward_id（新格式：
    iter_{iter}_sandbox_iter_{sandbox_iter}_response_{response_id}），
    在 (sci_logs, cri_logs) 里选择包含该 reward_id 的最佳一对：
      - 至少要 scientist 和 critic 两边都存在对应 run
      - 按 (sci_ep_cnt, cri_ep_cnt, last_sci_ep, last_cri_ep) 降序择优
    返回: (reward_id, best_sci_log, best_cri_log, best_index)
    """

    # 规范化：iter_001_sandbox_iter_07_response_0009 -> iter_1_sandbox_iter_7_response_9
    def _canon_new_rid_local(rid: str) -> str:
        m = re.fullmatch(r'(?i)iter_(\d+)_sandbox_iter_(\d+)_response_(\d+)', rid.strip())
        if not m:
            raise ValueError(f"Unrecognized new-format reward_id: {rid}")
        it, sb, resp = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"iter_{it}_sandbox_iter_{sb}_response_{resp}"

    # 宽松规范化：不符合新格式则返回 None（用于比较时跳过）
    def _canon_or_none_local(rid: str):
        try:
            return _canon_new_rid_local(rid)
        except Exception:
            return None

    m = re.search(
        r'(?i)Recommended function reward_id:\s*(iter_\d+_sandbox_iter_\d+_response_\d+)',
        llm_response
    )
    if not m:
        raise ValueError("No new-format reward_id found in llm_response "
                         "(expect 'iter_#_sandbox_iter_#_response_#').")

    rid = _canon_new_rid_local(m.group(1))

    candidates = []
    for i, (sci, cri) in enumerate(zip(sci_logs, cri_logs)):
        run_sci = next((r for r in getattr(sci, "_runs", [])
                        if _canon_or_none_local(getattr(r, "reward_id", "")) == rid), None)
        run_cri = next((r for r in getattr(cri, "_runs", [])
                        if _canon_or_none_local(getattr(r, "reward_id", "")) == rid), None)
        if run_sci is None or run_cri is None:
            continue

        sci_ep_cnt = len(getattr(run_sci, "sci_epochs", []))
        cri_ep_cnt = len(getattr(run_cri, "critic_epochs", []))
        last_sci_ep = max((e.epoch for e in getattr(run_sci, "sci_epochs", [])), default=0)
        last_cri_ep = max((e.epoch for e in getattr(run_cri, "critic_epochs", [])), default=0)

        candidates.append((i, sci, cri, sci_ep_cnt, cri_ep_cnt, last_sci_ep, last_cri_ep))

    if not candidates:
        raise ValueError(f"No logs contain reward_id '{rid}' (new format).")

    candidates.sort(key=lambda t: (t[3], t[4], t[5], t[6]), reverse=True)
    best_index, best_sci, best_cri, *_ = candidates[0]
    return rid, best_sci, best_cri, best_index



if __name__ == "__main__":
    # 简单单测：构造一个 InputLog，添加 run 和 epoch，并打印
    code_F1 = "def compute_rewards(seqs, amp_score, mic_score, meta_review_score):\n    return 0.7*amp_score + 0.2*mic_score + 0.1*meta_review_score"
    code_F2 = "def compute_rewards(seqs, amp_score, mic_score, meta_review_score):\n    return 0.6*amp_score + 0.3*mic_score + 0.1*meta_review_score"

    sci = InputLog(role="scientist")
    cri = InputLog(role="critic")

    sci.add_run("F1", code_F1)
    cri.add_run("F1", code_F1)

    for ep in range(1, 4):
        sci.add_epoch_scientist("F1", ep, mic_raw=10+ep, mic_score=0.1*ep, amp_score=0.5+0.1*ep, meta_review_score=0.6-0.05*ep, stage=str(ep-1))
        cri.add_epoch_critic("F1", ep, review_message=f"Round {ep}: looks OK.", stage=str(ep-1))

    # print("\n=== Scientist ===")
    # print(sci.to_str_scientist("F1"))

    # print("\n=== Critic ===")
    # print(cri.to_str_critic("F1"))

    # 基于 F1 分叉出 F2（独立容器）：不复制父的 epoch
    sci_F2 = sci.fork_as_independent("F1", "F2", new_code_str=code_F2, copy_epochs=False, include_parent=False)

    # print("\n=== Scientist: F2 only ===")
    # print(sci_F2.to_str_scientist("F2"))

    # print("\n=== Scientist: all runs in one container ===")
    # print(sci.to_str_scientist())

    # # Critic 部分
    # cri.add_run("F2", code_F2)

    # print("\n=== Critic: F1 ===")
    # print(cri.to_str_critic("F1"))

    # # print("\n=== Critic: F2 ===")
    # # print(cri.to_str_critic("F2"))

    # # 基于 F2 分叉出 F2a（独立容器）：同样不复制父的 critic 历史
    # cri_F2a = cri.fork_as_independent("F2", "F2a", new_code_str=code_F2a, copy_epochs=False, include_parent=True)

    # for ep in range(1, 4):
    #     cri_F2a.add_epoch_critic("F2a", ep, review_message=f"E{ep}: stronger AMP weight; check safety window.")

    # # print("\n=== Critic: F2a only ===")
    # # print(cri_F2a.to_str_critic("F2a"))

    print("\n=== Critic: parent + F2a in one container ===")
    # print(cri_F2a.to_str_critic())
