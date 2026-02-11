import json
from typing import List
import ast

# 1. Biopython ProteinAnalysis
from Bio.SeqUtils import ProtParam

def biopython_protein_analysis(seqs: List[str]) -> str:
    """
    使用Biopython的ProteinAnalysis分析多个蛋白质序列
    """
    if isinstance(seqs, str):
        try:
            seqs = ast.literal_eval(seqs)
        except (ValueError, SyntaxError):
            return [{'error': f"无法将字符串参数解析为列表: {seqs}"}]
            
    # print(f"[DEBUG] 调用函数: biopython_protein_analysis, 输入: {seqs}")
    print(f"[DEBUG] Function: biopython_protein_analysis, Input: {seqs}")
    results = []
    for seq in seqs:
        try:
            params = ProtParam.ProteinAnalysis(seq)
            for attr in dir(params):
                if attr.startswith("_"):  # 跳过私有属性
                    continue
                try:
                    obj = getattr(params, attr)
                    if callable(obj):  # 如果是方法
                        # 尝试调用无参数的方法
                        value = obj()
                    else:
                        value = obj
                    print(f"{attr}: {value}")
                except TypeError:
                    # 跳过需要传参的方法，如 charge_at_pH(pH)
                    pass
            result = {
                'sequence': seq,
                'molecular_weight': params.molecular_weight(),
                'aromaticity': params.aromaticity(),
                'instability_index': params.instability_index(),
                'flexibility': params.flexibility(),
                'isoelectric_point': params.isoelectric_point(),
                'gravy': params.gravy(),
                'charge_at_pH_7.4': params.charge_at_pH(7.4),
                'secondary_structure_fraction': params.secondary_structure_fraction(),
                'amino_acids_percent': params.get_amino_acids_percent(),
            }
            results.append(result)
        except Exception as e:
            results.append({'sequence': seq, 'error': str(e)})
    return json.dumps({"biopython_protein_analysis": results})  