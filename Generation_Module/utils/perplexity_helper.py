# -*- coding: utf-8 -*-
"""
Perplexity 报告核算器（p 加权版）

功能：
1) 读取你上传的 CoreWordList.json（位于 /mnt/data/CoreWordList.json）
2) 从 Perplexity 输出的报告文本中解析四个维度（EFF/Safe/DevStruct/Orig）的 [Tags: ...] 列表
3) 用“权重 × p”逐项累加，得到每个维度的正确分数
4) 将原文中的 [Score_XXX: ...] 替换为核算后的数值，返回“修正后的完整报告文本”
5) 同时返回一个字典，给出每个维度的数值明细

使用方式示例见文件末尾的 `if __name__ == "__main__":` 部分。
"""
import json
import re
import pathlib



# 维度名映射（报告中的标签 -> JSON中的维度分组名）
DIM_NAME_MAP = {
    "EFF": "Eff",
    "Safe": "Safe",
    "DevStruct": "DevStruct",
    "Orig": "Orig",
}

# 各维度对应的分数占位符名字（报告里用来替换）
SCORE_TOKEN_MAP = {
    "EFF": "Score_EFF",
    "Safe": "Score_Safe",
    "DevStruct": "Score_DS",
    "Orig": "Score_Orig",
}

# 解析 tag 的正则：(core=xxx, state=yyy, p=0.85)
TAG_RE = re.compile(
    r"\(core=([a-zA-Z0-9_]+)\s*,\s*state=([a-zA-Z0-9_]+)\s*,\s*p=([0-9.]+)\)"
)

def _fmt_score(x: float) -> str:
    """把分数格式化为 +0.123 的形式，保留三位小数。"""
    return f"{x:+.3f}"

def compute_dim_score(dim_key: str, tag_list_text: str, amp_generator_root) -> float:
    """
    给定维度（EFF/Safe/DevStruct/Orig）与该维度 [Tags: ...] 内的原始文本，
    解析出 (core, state, p) 并按“权重×p”累加。
    """
    CORE_PATH = pathlib.Path(f"{amp_generator_root}/utils/prompt/CoreWordList.json")

    # 读取权重表
    WEIGHTS = json.loads(CORE_PATH.read_text(encoding="utf-8"))
    dim_json_key = DIM_NAME_MAP[dim_key]
    score = 0.0
    unknowns = []
    for core, state, p_str in TAG_RE.findall(tag_list_text):
        p = float(p_str)
        weights_table = WEIGHTS[dim_json_key]["weights"]
        if core in weights_table and state in weights_table[core]:
            score += weights_table[core][state] * p
        else:
            unknowns.append((core, state))
    # 如有未知 core/state，打印提示（不终止），便于定位词表缺失或拼写问题
    if unknowns:
        print(f"[WARN] 未在权重表中找到 {dim_key} 维度的以下 core/state：{unknowns}")
    return score

import re
from typing import Tuple

_SECTION_HEAD_RE = re.compile(
    r'(?m)^\s*\[(?:EFF|Safe|DevStruct|Orig)\s*(?:[:：])?\]',
    re.IGNORECASE
)

def extract_model_block(report_text: str) -> Tuple[str, int, int]:
    if not report_text:
        return "", 0, 0
    m = _SECTION_HEAD_RE.search(report_text)
    start = m.start() if m else 0   # 找不到就退回全文
    end = len(report_text)
    return report_text[start:end], start, end


def replace_scores_in_perplexity(report_text: str,amp_generator_root) -> tuple[str, dict]:
    """
    对“Perplexity”模型段落：
      1) 解析四个维度的 [Tags: ...]
      2) 用 p 加权求和得到分数
      3) 替换对应的 [Score_XXX: ...] 数值
    返回：(修正后的全文, {dim: score})
    """
    block, s, e = extract_model_block(report_text)
    block_new = block
    results = {}

    # 逐维度处理
    for dim in ("EFF", "Safe", "DevStruct", "Orig"):
        # 找到该维度段落里的 [Tags: ...]，只在该段内搜索
        dim_re = re.compile(rf"\[{dim}:\][\s\S]*?\[Tags:\s*(.*?)\]", re.IGNORECASE)
        m = dim_re.search(block_new)
        if not m:
            print(f"[WARN] 未在 Perplexity 段落中找到 {dim} 的 [Tags: ...]。")
            continue
        tags_text = m.group(1)
        score = compute_dim_score(dim, tags_text, amp_generator_root)
        results[dim] = score

        # 替换该维度的 [Score_XXX: ...] 数值（只替换首次出现）
        token = SCORE_TOKEN_MAP[dim]
        score_re = re.compile(rf"\[{re.escape(token)}:\s*([^\]]*)\]")
        rep = f"[{token}: {_fmt_score(score)}]"
        block_new = score_re.sub(rep, block_new, count=1)

    # 回填回整份报告
    fixed_text = report_text[:s] + block_new + report_text[e:]
    return fixed_text, results

# ---------- 示例运行 ----------
if __name__ == "__main__":
    sample = """
🧠 Perplexity 正在评估...
[Perplexity]：
[EFF:][comment]本批10条肽均表现出极低的MIC预测值，均值远低于0.2 μg/mL，显示整体效力极佳；AMP概率多数较高，X4达0.94，X1和X3亦超0.7，支持强效抗菌活性；个别序列如X6、X7 AMP概率较低，但MIC仍佳，或有潜力非典型机制。理化性质中部分疏水性和净电荷平衡良好，有助于活性发挥。整体抗菌效力强，且疏水矩稳定，X4和X3表现尤佳。[Tags: (core=mic_band, state=low, p=1.00) | (core=amp_likelihood, state=high, p=0.85) | (core=hydrophobicity, state=balanced, p=0.60)][Score_EFF: +1.37][notes] MIC极低且AMP概率与理化性质相互佐证，提高判定置信度
[Safe:][comment]毒性预测分布广，X4毒性最高达1.0，明显风险；X2与X3等中等毒性偏高，需警惕；部分如X6、X7、X9、X10毒性极低，具更佳安全性；无明显过度阳离子化与芳香富集信号，整体安全性受高毒性候选拖累。需关注X4的高毒风险并优先排查。[Tags: (core=toxinpred, state=medium, p=0.85) | (core=toxinpred, state=high, p=0.60)][Score_Safe: -0.07][notes] 高毒性肽X4严重拉低整体安全评分
[DevStruct:][comment]结构置信度pLDDT总体较高，多数超过0.7，X4最高达0.91，结构预测稳定可靠；部分肽不稳定指数较高（如X6、X9），潜在降解风险需进一步评估；长度多在适中带范围，有利开发；理化查询显示疏水性适中，提示溶解性尚可，X1、X3、X4表现尤佳。总体开发可行性较好。[Tags: (core=plddt, state=high, p=0.85) | (core=instability_index, state=medium, p=0.60) | (core=length_band, state=optimal, p=0.60)][Score_DS: +0.63][notes] 高pLDDT与适中不稳定指数联合支持较好结构可靠性
[Orig:][comment]与模板肽相似度普遍中高，均分约0.55以上，存在较多结构复用，创新性受限；批内多样性中等，重复度存在；低复杂度成分未见显著偏高，说明序列复杂性尚可；建议进一步优化多样性以提升原创价值。[Tags: (core=foldseek_similarity, state=medium, p=1.00) | (core=batch_diversity, state=medium, p=0.85)][Score_Orig: +0.37][notes] 模板复用率较高限制了原创性提升
"""
    fixed, results = replace_scores_in_perplexity(sample)
    print("【核算结果】", {k: round(v, 6) for k, v in results.items()})
    print("\n【修正后的报告】：\n")
    print(fixed)
