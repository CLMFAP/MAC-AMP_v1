# -*- coding: utf-8 -*-
"""
Robust Meta-Scoring helper for Area Chair formatting.

This version is tolerant to:
- Missing or placeholder Dist blocks (e.g., "[Dist: none][Num: 0]")
- Reviewers without [Score_*] (they are ignored instead of defaulting to 0.1)
- Only 1 or 2 reviewers providing scores (no disagreement penalty applied)
- 3 or more reviewers providing scores (use Dist-based penalty as before)

Public API:
- compute_metacore_dimension(reviewers, area_meta, dim)
- compute_metacore_all(reviewers, area_meta)
- compute_meta_score(all_results)  -> overall score + components
- meta_score_from_all(all_results) -> alias of compute_meta_score
- format_area_meta_with_scores(reviewers, area_meta) -> str (rewritten AC text)

Dimensions supported: EFF, Safe, DevStruct (alias DS), Orig
Score tags supported:  [Score_EFF], [Score_Safe], [Score_DS] or [Score_DevStruct], [Score_Orig]

Example usage is provided in __main__.
"""
from __future__ import annotations
import re
from typing import Dict, List, Any, Tuple, Optional

# -----------------------------
# Aliases & constants
# -----------------------------
_ALIAS_BLOCK: Dict[str, List[str]] = {
    "EFF": ["EFF"],
    "Safe": ["Safe"],
    "DevStruct": ["DevStruct", "DS"],
    "Orig": ["Orig"],
}

_ALIAS_SCORE: Dict[str, List[str]] = {
    "EFF": ["EFF"],
    "Safe": ["Safe"],
    "DevStruct": ["DS", "DevStruct"],
    "Orig": ["Orig"],
}

_DIM_NORMALIZE = {
    "eff": "EFF",
    "safe": "Safe",
    "devstruct": "DevStruct",
    "ds": "DevStruct",
    "orig": "Orig",
}

__all__ = [
    "compute_metacore_dimension",
    "compute_metacore_all",
    "compute_meta_score",
    "meta_score_from_all",
    "format_area_meta_with_scores",
]

# -----------------------------
# Utilities
# -----------------------------

def _clip_unit(x: float) -> float:
    """Clip to [-1, 1]."""
    if x > 1.0:
        return 1.0
    if x < -1.0:
        return -1.0
    return x


def _parse_dist_for_dim(area_meta: str, dim: str) -> List[List[int]]:
    """Extract Dist groups for a given dimension from Area Chair text.

    Robust to missing Dist or placeholders like 'none'. Returns [] when unavailable.
    The Dist format supports arbitrary number of parenthesized integer groups, e.g.:
        [Dist: (1,0,-1)(0)(-1,-1,1)]
    """
    block = "|".join(map(re.escape, _ALIAS_BLOCK[dim]))
    m = re.search(
        rf"\[(?:{block}):\][\s\S]*?\[Dist:\s*([^\]]+?)\]\s*\[Num:\s*\d+\]",
        area_meta,
        flags=re.IGNORECASE,
    )
    if not m:
        # No Dist section found for this dimension
        return []
    dist_str = m.group(1).strip()
    # 'none' / 'null' / 'n/a' / '' treated as absent
    if not dist_str or re.fullmatch(r"none|null|n/?a", dist_str, flags=re.IGNORECASE):
        return []

    groups = re.findall(r"\(([^)]*)\)", dist_str)
    dist_groups: List[List[int]] = []
    for g in groups:
        xs = [x.strip() for x in g.split(",")]
        nums = [int(x) for x in xs if x and re.fullmatch(r"-?\d+", x)]
        if nums:
            dist_groups.append(nums)
    return dist_groups


def _penalty_from_groups(groups: List[List[int]]) -> Tuple[float, List[float]]:
    """Compute a disagreement penalty from Dist groups.

    Rule per original implementation: for each group, take mean m, then group_value = 1 - |m|.
    Penalty weight = average of group_values (0 if no groups).
    Returns (penalty_weight, per_group_values).
    """
    if not groups:
        return 0.0, []
    per_vals: List[float] = []
    for g in groups:
        if not g:
            continue
        m = sum(g) / float(len(g))
        per_vals.append(1.0 - abs(m))
    if not per_vals:
        return 0.0, []
    penalty = sum(per_vals) / float(len(per_vals))
    return penalty, per_vals


def _extract_score_from_text(text: str, dim: str) -> Optional[float]:
    """Grab [Score_*] for a dimension (supports DS/DevStruct alias).

    Returns None if not found to ensure non-responders don't affect averages.
    """
    for tag in _ALIAS_SCORE[dim]:
        m = re.search(rf"\[Score_{re.escape(tag)}:\s*([+-]?\d+(?:\.\d+)?)\]", text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


# -----------------------------
# Core computations
# -----------------------------

def compute_metacore_dimension(
    reviewers: Dict[str, str],
    area_meta: str,
    dim: str = "EFF",
) -> Dict[str, Any]:
    """Compute a dimension's Metacore with robust handling:

    - Normalize dim to {EFF, Safe, DevStruct, Orig}
    - Use only reviewers that actually provide [Score_*] for this dim.
    - If 0 responders -> meta_score = 0.0, penalty = 0, gamma = 1.
    - If 1 or 2 responders -> meta_score = mean of responders, penalty = 0, gamma = 1.
    - If >=3 responders -> meta_score = mean of responders; penalty computed from Dist groups (if any),
      gamma = clip[0.6,1.0](1 - 0.6 * penalty_weight).
    """
    dim_norm = _DIM_NORMALIZE.get(dim.lower(), dim)
    if dim_norm not in _ALIAS_BLOCK:
        raise ValueError(f"Unsupported dimension: {dim}")

    # Collect scores only from reviewers who provided explicit [Score_*]
    scores: List[float] = []
    for name, text in reviewers.items():
        s = _extract_score_from_text(text, dim_norm)
        if s is not None:
            scores.append(float(s))
    num_responders = len(scores)
    meta_score = (sum(scores) / float(num_responders)) if num_responders > 0 else 0.0

    # Disagreement penalty logic
    dist_groups = _parse_dist_for_dim(area_meta, dim_norm)
    if num_responders >= 3:
        penalty_weight, group_values = _penalty_from_groups(dist_groups)
        gamma = max(0.6, min(1.0, 1.0 - 0.6 * penalty_weight))
    else:
        penalty_weight, group_values = (0.0, [] if not dist_groups else [0.0] * len(dist_groups))
        gamma = 1.0

    metacore_val = gamma * meta_score
    return {
        "dimension": dim_norm,
        "penalty_weight": penalty_weight,
        "meta_score": meta_score,
        "Metacore": metacore_val,
        "debug": {
            "num_responders": num_responders,
            "group_values": group_values,
            "dist_groups": dist_groups,
            "reviewer_scores": scores,
            "gamma": gamma,
        },
    }


def compute_metacore_all(reviewers: Dict[str, str], area_meta: str) -> Dict[str, Dict[str, Any]]:
    """Compute per-dimension results for all four dimensions."""
    results: Dict[str, Dict[str, Any]] = {}
    for d in ("EFF", "Safe", "DevStruct", "Orig"):
        results[d] = compute_metacore_dimension(reviewers, area_meta, d)
    return results


def compute_meta_score(all_results: Dict[str, Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    """Compute the overall MetaScore as the mean of per-dimension Metacore values.

    Returns (overall_meta, components), where components contains the four per-dimension Metacore values.
    """
    comps = {
        d: float(all_results[d]["Metacore"]) if d in all_results else 0.0
        for d in ("EFF", "Safe", "DevStruct", "Orig")
    }
    overall = _clip_unit(sum(comps.values()) / 4.0)
    return overall, comps


# Backward-compatible alias
meta_score_from_all = compute_meta_score


# -----------------------------
# Formatting / rewriting Area Chair text
# -----------------------------

def _rewrite_block_with_score(block_text: str, dim: str, val: float) -> str:
    """Remove [Dist:*][Num:*] from a dimension block and append [MetaScore_dim: +x.xxx]."""
    # Remove [Dist: ...]
    block_text = re.sub(r"\[Dist:\s*[^\]]*\]\s*", "", block_text)
    # Remove [Num: ...]
    block_text = re.sub(r"\[Num:\s*\d+\]\s*", "", block_text)
    # Append meta score tag (space before to keep blocks compact, but distinct)
    return block_text.rstrip() + f"\n[MetaScore_{dim}: {val:+.3f}]\n"


def format_area_meta_with_scores(reviewers: Dict[str, str], area_meta: str) -> str:
    """Rewrite Area Chair meta text by injecting [MetaScore_*] tags and removing Dist/Num.

    - Computes per-dimension Metacore using the robust rules above
    - For each dimension block, strips [Dist] and [Num]
    - Appends [MetaScore_DIM: +/-x.xxx] to each block
    - Appends overall [MetaScore: +/-x.xxx] at the end
    """
    all_res = compute_metacore_all(reviewers, area_meta)
    overall, comps = compute_meta_score(all_res)

    # For each dimension, find its block and rewrite
    rewritten = area_meta
    for dim in ("EFF", "Safe", "DevStruct", "Orig"):
        alias = "|".join(map(re.escape, _ALIAS_BLOCK[dim]))
        # capture the whole block lazily up to next [X:] or end
        pattern = rf"(\[(?:{alias}):\][\s\S]*?)(?=(\n\[[A-Za-z]+:?|$))"
        def _repl(m: re.Match) -> str:
            block_text = m.group(1)
            return _rewrite_block_with_score(block_text, dim, comps[dim])
        rewritten = re.sub(pattern, _repl, rewritten, flags=re.IGNORECASE)

    # Append overall MetaScore at the end if not already present
    if not re.search(r"\[MetaScore:\s*[+-]?\d+(?:\.\d+)?\]", rewritten):
        rewritten = rewritten.rstrip() + f"\n[MetaScore: {overall:+.3f}]\n"
    return rewritten


# -----------------------------
# Demo (optional)
# -----------------------------
if __name__ == "__main__":
    reviewers_demo = {
        "GenAI": (
            "[EFF:][comment]None[Tags: None][notes]None[Dist: (0)] [Num: 1]\n"
            "[Safe:][comment]None[Tags: None][notes]None[Dist: (0)] [Num: 1]\n"
            "[DevStruct:][comment]None[Tags: None][notes]None[Dist: (0)] [Num: 1]\n"
            "[Orig:][comment]None[Tags: None][notes]None[Dist: (0)] [Num: 1]\n"
        ),
        "Perplexity": (
            "[EFF:][comment]... [Score_EFF: +1.078]\n"
            "[Safe:][comment]... [Score_Safe: -0.170]\n"
            "[DevStruct:][comment]... [Score_DS: +0.545]\n"
            "[Orig:][comment]... [Score_Orig: -0.055]\n"
        ),
        "OpenAI": (
            "[EFF:][comment]None[Tags: None][notes]None[Dist: (0)] [Num: 1]\n"
            "[Safe:][comment]None[Tags: None][notes]None[Dist: (0)] [Num: 1]\n"
            "[DevStruct:][comment]None[Tags: None][notes]None[Dist: (0)] [Num: 1]\n"
            "[Orig:][comment]None[Tags: None][notes]None[Dist: (0)] [Num: 1]\n"
        ),
    }

    area_meta_demo = (
        "[EFF:][meta_comment]... [EFF_Tags: ...][Dist: none][Num: 0]\n"
        "[Safe:][meta_comment]... [Safe_Tags: ...][Dist: none][Num: 0]\n"
        "[DevStruct:][meta_comment]... [DS_Tags: ...][Dist: none][Num: 0]\n"
        "[Orig:][meta_comment]... [Orig_Tags: ...][Dist: none][Num: 0]\n"
    )

    all_res = compute_metacore_all(reviewers_demo, area_meta_demo)
    overall, comps = compute_meta_score(all_res)

    print("Per-dimension results:")
    for k, v in all_res.items():
        print(k, v)
    print("Overall:", overall, comps)

    formatted = format_area_meta_with_scores(reviewers_demo, area_meta_demo)
    print("\nRewritten Area Meta:\n", formatted)
