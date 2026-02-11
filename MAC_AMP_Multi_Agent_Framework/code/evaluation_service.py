import json
from typing import List, Dict, Any
import time
import os
import subprocess
import gzip
import pickle
import glob
import pandas as pd
import numpy as np
import ast
from Bio.PDB import PDBParser
import logging
logging.basicConfig(level=logging.ERROR)
import torch
from evaluation_biopython import biopython_protein_analysis
import math
import re
import tempfile
import sys
from utils.mic_hill_mapping import hill_score_m0_1_a1_from_logmic

def foldseek_similarity_score(evalue: float,
                              alntmscore: float,
                              fident: float,   # 0..1（easy-search 输出即小数）
                              qcov: float=None, # 0..1，可选
                              tcov: float=None  # 0..1，可选
                              ) -> float:
    """
    短肽友好版综合分：
    - alntmscore：用 tanh 压缩，避免过早饱和
    - fident：直接用 0..1
    - evalue：对数缩放（越小越接近 1）
    - coverage：qcov/tcov 二者最小值，弱权重防止局部对齐刷高分
    """
    # 1) 结构分（建议把 2.0 当作“高相似”的参考点）
    score_norm = math.tanh(alntmscore) / math.tanh(2.0)   # ≈0..1，1~2 之间仍有区分度
    score_norm = max(0.0, min(score_norm, 1.0))
    # 2) 序列一致性（0..1），保持不变
    fident_norm = max(0.0, min(float(fident), 1.0))
    # 3) 覆盖度（可选）
    if qcov is not None and tcov is not None:
        cov = max(0.0, min(min(float(qcov), float(tcov)), 1.0))
    else:
        cov = 1.0  # 缺列就不惩罚
    # 4) e-value 对数缩放：e<=1e-6 ~1；e>=1e+2 ~0
    if evalue is None or evalue <= 0:
        e_bonus = 1.0
    else:
        x = -math.log10(evalue)           # e=1e-6->6，e=1->0，e=100->-2
        e_bonus = (x + 2) / 8             # [-2,6] -> [0,1]
        e_bonus = max(0.0, min(e_bonus, 1.0))
    # 5) 权重（短肽：序列一致性更重要些）
    w_score, w_fident, w_cov, w_eval = 0.35, 0.45, 0.10, 0.10
    final = (w_score * score_norm +
             w_fident * fident_norm +
             w_cov   * cov +
             w_eval  * e_bonus)

    return round(final, 3)

def foldseek_compare(
    seqs: List[str],
    ref_pdb_path: str,
    tmp_dir: str = "tmp_foldseek"
) -> List[float]:
    """
    使用 OmegaFold 预测每条 query 的结构，然后用 foldseek easy-search
    将 query_pdb 与 ref_pdb_path（可为目录或单PDB）进行结构比对。
    返回每条 query 的 similarity_score 列表。
    """
    if isinstance(seqs, str):
        try:
            seqs = ast.literal_eval(seqs)
        except (ValueError, SyntaxError):
            return [0.0]

    print(f"[DEBUG] Function: foldseek_compare(easy-search), Input N={len(seqs)}")
    os.makedirs(tmp_dir, exist_ok=True)

    def _clean_seq(s: str) -> str:
        s = s.strip().upper()
        return (s.replace("X", "G")
                 .replace("U", "C")
                 .replace("Z", "E")
                 .replace("J", "L")
                 .replace("B", "D")
                 .replace("O", "K"))

    results = []
    for idx, query_seq in enumerate(seqs):
        record = {
            'sequence': query_seq,
            'status': None,
            'result_file': None,
            'preview': None,
            'error': None,
            'top_hit': None,
            'score': None,     # 将填入 alntmscore
            'evalue': None,
            'fident': None,
            'similarity_score': None
        }

        # Step 1: 写 FASTA（对非常见氨基酸做轻量替换以保证 OmegaFold 可运行）
        fasta_path = os.path.join(tmp_dir, f"query_{idx}.fa")
        pdb_out_dir = os.path.join(tmp_dir, f"query_pdb_{idx}")
        os.makedirs(pdb_out_dir, exist_ok=True)
        cleaned = _clean_seq(query_seq)
        with open(fasta_path, 'w') as f:
            f.write(f">query{idx}\n{cleaned}\n")
        # Step 2: 运行 OmegaFold 生成 query PDB
        try:
            _ = subprocess.run(
                ["omegafold", "--model", "2", fasta_path, pdb_out_dir],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ OmegaFold 出错（idx={idx}）: {query_seq}")
            print("⛔ stderr:\n", (e.stderr or "").strip()[:500])
            print("📤 stdout:\n", (e.stdout or "").strip()[:500])
            record['status'] = 'omegafold_failed'
            record['error'] = e.stderr or str(e)
            results.append(record)
            continue

        pdb_files = glob.glob(os.path.join(pdb_out_dir, "*.pdb"))
        if not pdb_files:
            record['status'] = 'no_structure_generated'
            record['error'] = 'OmegaFold未生成任何PDB结构文件'
            results.append(record)
            continue
        query_pdb = pdb_files[0]
        # 释放显存（可选）
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
        # Step 3: 用 foldseek easy-search（直接产出 TSV）
        tsv_path = os.path.join(tmp_dir, f"foldseek_result_{idx}.tsv")
        try:
            # ref_pdb_path 可以是目录或单个 PDB 文件
            proc = subprocess.run(
                ["foldseek", "easy-search",
                 query_pdb,
                 ref_pdb_path,
                 tsv_path,
                 tmp_dir,
                 "--threads", "2",
                 "--format-output", "query,target,evalue,alntmscore,fident,qcov,tcov",
                 "--exhaustive-search", "1",
                 "-s", "12",
                 "--max-seqs", "10000",
                 "--prefilter-mode", "1"],
                check=True,
                capture_output=True,
                text=True
            )
            if proc.stdout:
                print(f"[foldseek stdout idx={idx}] {proc.stdout[:2000]}")
            if proc.stderr:
                print(f"[foldseek stderr idx={idx}] {proc.stderr[:2000]}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Foldseek easy-search 出错（idx={idx}）: {query_seq}")
            print("⛔ stderr:\n", (e.stderr or "").strip()[:2000])
            print("📤 stdout:\n", (e.stdout or "").strip()[:2000])
            record['status'] = 'foldseek_failed'
            record['error'] = e.stderr or str(e)
            results.append(record)
            continue
        # Step 4: 解析 TSV 结果（取 top-1）
        try:
            if (not os.path.exists(tsv_path)) or (os.path.getsize(tsv_path) == 0):
                record['status'] = 'no_match'
                record['result'] = 'No significant similarity found.'
                results.append(record)
                continue

            with open(tsv_path, 'r') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            if not lines:
                record['status'] = 'no_match'
                record['result'] = 'No significant similarity found.'
                results.append(record)
                continue

            record['preview'] = lines[:5]
            cols = lines[0].split('\t')  # query, target, evalue, alntmscore, fident
            if len(cols) >= 5:
                record['top_hit'] = cols[1]
                record['evalue'] = float(cols[2]) if cols[2] not in ("", "inf", "nan") else 1.0
                record['score']  = float(cols[3]) if cols[3] not in ("", "inf", "nan") else 0.0
                record['fident'] = float(cols[4]) if cols[4] not in ("", "inf", "nan") else 0.0
                record['similarity_score'] = foldseek_similarity_score(
                    record['evalue'], record['score'], record['fident']
                )
                record['status'] = 'success'
                record['result_file'] = tsv_path
            else:
                record['status'] = 'success_partial'
                record['error'] = f'输出列数不足，行内容: {lines[0]}'
                record['result_file'] = tsv_path
        except Exception as e:
            record['status'] = 'success_partial'
            record['result_file'] = tsv_path
            record['error'] = f'解析TSV失败: {e}'
            print(f'foldseek_compare(easy-search) 解析失败: {e}')
        results.append(record)

    # print(f'[DEBUG] foldseek_compare(easy-search) 结果条目: {results}')
    similarity_scores = [r['similarity_score'] if r['similarity_score'] is not None else 0.0 for r in results]
    print(f'[DEBUG] foldseek_compare(easy-search) 结果条目: {similarity_scores}')
    return similarity_scores

# 3. Macrel抗菌肽预测
def macrel_predict(seqs: List[str], model_path: str = "test_macrel/AMP.pkl.gz") -> str:
    """
    用Macrel模型预测多个抗菌肽分数
    """
    if isinstance(seqs, str):
        try:
            seqs = ast.literal_eval(seqs)
        except (ValueError, SyntaxError):
            return [{'error': f"无法将字符串参数解析为列表: {seqs}"}]
            
    # print(f"[DEBUG] 调用函数: macrel_predict, 输入: {seqs}")
    print(f"[DEBUG] Function: macrel_predict, Input: {seqs}")
    try:
        import sys
        sys.path.append("test_macrel")
        from predictor import macrel_predictor
        model = pickle.load(gzip.open(model_path, 'rb'))
        scores = macrel_predictor(seqs, model)
        results = [{'sequence': seq, 'score': float(score)} for seq, score in zip(seqs, scores)]
        # return json.dumps({"macrel_predict": results})
        scores_list = [r['score'] if 'score' in r else -1.0 for r in results]
        print(f"[DEBUG] Function: macrel_predict 结果条目: {scores_list}")
        return scores_list
    except Exception as e:

        results = [{'sequence': seq, 'error': str(e)} for seq in seqs]
        return json.dumps(results)
    
def extract_plddt_from_pdb(pdb_file: str) -> float:
    """
    从PDB文件中提取平均plDDT值（假设plDDT存储在B-factor字段）
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("predicted", pdb_file)
    b_factors = [
        atom.get_bfactor()
        for model in structure
        for chain in model
        for residue in chain
        for atom in residue
    ]
    if b_factors:
        return sum(b_factors) / len(b_factors)
    else:
        return -1.0  # 表示未能获取有效plDDT
# 4. OmegaFold结构预测

def omegafold_predict(seqs: List[str], tmp_dir: str = "tmp_omegafold") -> str:
    """
    用 OmegaFold 预测多个结构，返回每个 PDB 文件路径。
    在失败时仅打印错误信息，正常运行时不打印任何日志。
    """
    if isinstance(seqs, str):
        try:
            seqs = ast.literal_eval(seqs)
        except (ValueError, SyntaxError):
            return [{'error': f"无法将字符串参数解析为列表: {seqs}"}]

    os.makedirs(tmp_dir, exist_ok=True)
    results = []

    for idx, seq in enumerate(seqs):
        fasta_path = os.path.join(tmp_dir, f"query_{idx}.fa")
        pdb_out_dir = os.path.join(tmp_dir, f"query_pdb_{idx}")

        with open(fasta_path, 'w') as f:
            f.write(f">query{idx}\n{seq}\n")

        command = ["omegafold", "--model", "2", fasta_path, pdb_out_dir]

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ OmegaFold 出错（序列 idx={idx}）: {seq}")
            print("⛔ stderr:")
            print(e.stderr.strip())
            print("📤 stdout:")
            print(e.stdout.strip())
            results.append({'sequence': seq, 'error': f'OmegaFold failed with return code {e.returncode}'})
            continue
        except Exception as e:
            results.append({'sequence': seq, 'error': f'OmegaFold exception: {e}'})
            continue

        pdb_files = glob.glob(os.path.join(pdb_out_dir, "*.pdb"))
        if not pdb_files:
            results.append({'sequence': seq, 'error': 'No PDB file generated by OmegaFold'})
            continue

        pdb_file = pdb_files[0]
        plddt_score = extract_plddt_from_pdb(pdb_file)
        results.append({
            'sequence': seq,
            'pdb_file': pdb_file,
            'plddt': round(plddt_score/100.0, 4)
        })
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # return json.dumps({"omegafold_predict": results}, indent=2)
    plddts = [r['plddt'] if 'plddt' in r else 0.0 for r in results]
    print(f"[DEBUG] Function: omegafold_predict 结果条目: {plddts}")
    return plddts

# 5. ToxinPred3毒性预测
def toxinpred3_predict(seqs: List[str], model: int = 2, threshold: float = 0.38) -> str:
    """
    用toxinpred3命令行预测多个序列的毒性，所有序列一次性写入同一个FASTA文件，批量预测。
    若输入序列过短，自动补充一条标准长肽，保证特征提取不报错。
    """
    if isinstance(seqs, str):
        try:
            seqs = ast.literal_eval(seqs)
        except (ValueError, SyntaxError):
            return [{'error': f"无法将字符串参数解析为列表: {seqs}"}]
            
    # print(f"[DEBUG] 调用函数: toxinpred3_predict, 输入: {seqs}")
    print(f"[DEBUG] Function: toxinpred3_predict, Input: {seqs}")
    # 若有短序列，补充一条标准肽，避免特征提取报错
    seqs_for_pred = list(seqs)
    need_dummy = any(len(seq) < 10 for seq in seqs)
    if need_dummy:
        seqs_for_pred.append("KLFKFFKDLLGKFLG")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as input_file:
        for i, s in enumerate(seqs_for_pred):
            input_file.write(f">seq{i}\n{s}\n")
        input_path = input_file.name
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as output_file:
        output_path = output_file.name
    try:
        cmd = f"toxinpred3 -i {input_path} -o {output_path} -m {model} -t {threshold}"
        subprocess.run(cmd, shell=True, check=True)
        df = pd.read_csv(output_path)
        results = []
        for i, seq in enumerate(seqs):
            # 若df行数不足，返回空dict
            if i < len(df):
                result = df.iloc[i].to_dict()
            else:
                result = {}
            result['sequence'] = seq
            results.append(result)
        # return json.dumps({"toxinpred3_predict": results}   )
        scores = [r.get("Hybrid Score", -1.0) for r in results]
        print(f"[DEBUG] Function: toxinpred3_predict 结果条目: {scores}")
        return scores

    except Exception as e:
        print("Command failed!")
        if hasattr(e, 'returncode'):
            print("Return code:", e.returncode)
        if hasattr(e, 'cmd'):
            print("Command:", e.cmd)
        if hasattr(e, 'stdout'):
            print("stdout:", e.stdout)
        if hasattr(e, 'stderr'):
            print("stderr:", e.stderr)
        results = [{'sequence': seq, 'error': str(e)} for seq in seqs]
        return json.dumps({"toxinpred3_predict": results})
    finally:
        os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


def predict_mic(sequences: List[str],mic_regression_root,default_sign) -> List[float]:
    """
    使用训练好的 MIC 模型预测 log10(MIC) 值，并将其归一化为 [0,1] 的活性分数。
    越小代表 MIC 越低（抗菌活性越强），归一化后得分越高。
    """
    sys.path.append(mic_regression_root)
    
    try:
        from predict import predict_mic
    except ImportError as e:
        print(f"[错误] 导入 PredictMIC 失败: {e}")
        return [0.0] * len(sequences)

    try:
        log_mics = predict_mic(sequences, "EC", mic_regression_root,default_sign)
        print(log_mics)
        # 映射 log10(MIC) ∈ [-2, 4] 到 score ∈ [0, 1]
        # mic_scores = [(4.0 - min(max(m, -2.0), 4.0)) / 6.0 for m in log_mics]
        mic = [10**m for m in log_mics]
        print(f"[DEBUG] Function: predict_mic 结果条目: {mic}")
        return mic
    except Exception as e:
        print(f"[错误] MIC 模型预测失败: {e}")
        return [0.0] * len(sequences)
    
def predict_mic_score(sequences: List[str], mic_regression_root, default_sign) -> List[float]:
    sys.path.append(mic_regression_root)
    
    try:
        from predict import predict_mic
    except ImportError as e:
        print(f"[错误] 导入 PredictMIC 失败: {e}")
        return [0.0] * len(sequences)

    try:
        log_mics = predict_mic(sequences, "EC", mic_regression_root, default_sign)
        print("[DEBUG] log_mics:", log_mics)

        mic_scores = hill_score_m0_1_a1_from_logmic(log_mics, default_sign)
        # mic_scores = np.asarray(mic_scores, dtype=float).tolist()

        print(f"[DEBUG] Function: predict_mic_score 结果条目: {mic_scores}")
        return mic_scores
    except Exception as e:
        print(f"[错误] MIC 模型预测失败: {e}")
        return [0.0] * len(sequences)

# class Evaluator():
#     def __init__(self,use_default_reward,amp_generator_root,mic_regression_root,default_sign,workspace_dir ):
#         self.use_default_reward = use_default_reward
#         self.amp_generator_root=amp_generator_root
#         self.mic_regression_root=mic_regression_root
#         self.default_sign=default_sign
#         self.workspace_dir=workspace_dir
class Evaluator():
    def __init__(
        self,
        use_default_reward,
        amp_generator_root,
        mic_regression_root,
        default_sign,
        workspace_dir,
        ablation_mode: int = 0,
    ):
        self.use_default_reward = use_default_reward
        self.amp_generator_root = amp_generator_root
        self.mic_regression_root = mic_regression_root
        self.default_sign = default_sign
        self.workspace_dir = workspace_dir

        # 记录消融模式
        self.ablation_mode = ablation_mode

        # Va: ToxinPred3.0     Vb: OmegaFold
        # Sb: Macrel 对 RL reward 的贡献（后面单独在 reward 函数里处理）
        self.disable_va = ablation_mode in (2, 4, 6, 7)  # 需要把 Va 设为 N/A
        self.disable_vb = ablation_mode in (1, 4, 5, 7)  # 需要把 Vb 设为 N/A
        self.manual_reward_no_sb = ablation_mode in (3, 5, 6, 7)  # Sb 消融 → reward=(Sa+Sc)/2
        print(f"disable_va: {self.disable_va}")
        print(f"disable_vb: {self.disable_vb}")
        print(f"manual_reward_no_sb: {self.manual_reward_no_sb}")

            
    def run_evaluation_tools(self, sequences: List[str], ref_pdb_path: str = "/data/AMP_Escherichia_coli_811_new/ref_pdbs") -> Dict:
        print(f"--- [Evaluation Agent]：Evaluation Start - {len(sequences)} Sequences ---")
        timings = {}
        ref_pdb_path=self.amp_generator_root+ref_pdb_path
        tmp_foldseek=self.amp_generator_root+"/tmp_foldseek"
        macrel_model_path=self.amp_generator_root+"/test_macrel/AMP.pkl.gz"
        tmp_omegafold=self.amp_generator_root+"/tmp_omegafold"
        if self.workspace_dir:
            tmp_omegafold=self.workspace_dir+"/tmp_omegafold"

        try:
            start = time.time()
            phys_chem_results = json.loads(biopython_protein_analysis(seqs=sequences)).get("biopython_protein_analysis", [])
            timings['Biopython'] = time.time() - start
        except Exception as e:
            print(f"[警告] Biopython分析失败: {e}")
            phys_chem_results = [{"error": str(e)}] * len(sequences)

        # --- Va: ToxinPred3.0 ---
        if self.disable_va:
            print("[Ablation] Va (ToxinPred3.0) 被关闭，toxicity_results 设为 'N/A'")
            toxicity_results = ["N/A"] * len(sequences)
            timings['ToxinPred3'] = 0.0
        else:
            try:
                start = time.time()
                toxicity_results = toxinpred3_predict(seqs=sequences)
                timings['ToxinPred3'] = time.time() - start
            except Exception as e:
                print(f"[警告] ToxinPred3预测失败: {e}")
                toxicity_results = [-1.0] * len(sequences)


        try:
            start = time.time()
            antimicrobial_results = macrel_predict(seqs=sequences, model_path=macrel_model_path)
            timings['Macrel'] = time.time() - start
        except Exception as e:
            print(f"[警告] Macrel预测失败: {e}")
            antimicrobial_results = [-1.0] * len(sequences)

        # --- Vb: OmegaFold ---
        if self.disable_vb:
            print("[Ablation] Vb (OmegaFold) 被关闭，structure_results 设为 'N/A'")
            structure_results = ["N/A"] * len(sequences)
            timings['OmegaFold'] = 0.0
        else:
            try:
                start = time.time()
                structure_results = omegafold_predict(seqs=sequences, tmp_dir=tmp_omegafold)
                timings['OmegaFold'] = time.time() - start
            except Exception as e:
                print(f"[警告] OmegaFold预测失败: {e}")
                structure_results = [-1.0] * len(sequences)


        try:
            start = time.time()
            similarity_results = foldseek_compare(seqs=sequences, ref_pdb_path=ref_pdb_path, tmp_dir=tmp_foldseek)
            timings['Foldseek'] = time.time() - start
        except Exception as e:
            print(f"[警告] Foldseek比较失败: {e}")
            similarity_results = [-1.0] * len(sequences)


        try:
            start = time.time()
            mic_score = predict_mic_score(sequences,self.mic_regression_root,self.default_sign)
            timings['MIC_Predict'] = time.time() - start
        except Exception as e:
            print(f"[警告] MIC预测失败: {e}")
            mic_score = [-1.0] * len(sequences)

        try:
            start = time.time()
            mic_original = predict_mic(sequences,self.mic_regression_root,self.default_sign)
            timings['MIC_Predict'] = time.time() - start
        except Exception as e:
            print(f"[警告] MIC预测失败: {e}")
            mic_original = [-1.0] * len(sequences)

        # 最终确保所有结果都在 [0,1]，避免 reward 崩溃
        def clip_list(values: List[float]) -> List[float]:
            return [min(max(v, 0.0), 1.0) for v in values]

        # toxicity_results = clip_list(toxicity_results)
        # antimicrobial_results = clip_list(antimicrobial_results)
        # structure_results = clip_list(structure_results)
        # similarity_results = clip_list(similarity_results)
        # mic_results = clip_list(mic_results)

        combined_results = {
            "peptide_sequence": sequences,
            "physicochemical_properties": phys_chem_results,
            "toxicity": toxicity_results,
            "antimicrobial_activity": antimicrobial_results,
            "structure_prediction": structure_results,
            "similarity_analysis": similarity_results,
            "mic_prediction": mic_score,
            "mic_score": mic_score,
            "mic_original": mic_original
        }

        print(f"--- [Evaluation Agent]：Completed ---")
        return {"evaluation_result": combined_results, "timings": timings}

    def get_evaluation_outputs_from_agent(self, sequences: List[str]) -> Dict:
        final_message = self.run_evaluation_tools(sequences)
        evaluation_result = final_message["evaluation_result"]
        physicochemical_properties = evaluation_result["physicochemical_properties"]
        toxicity = evaluation_result["toxicity"]
        antimicrobial_activity = evaluation_result["antimicrobial_activity"]
        structure_prediction = evaluation_result["structure_prediction"]
        similarity_analysis = evaluation_result["similarity_analysis"]
        mic_score = evaluation_result["mic_score"]
        mic_original = evaluation_result["mic_original"]
        try:
            return evaluation_result, physicochemical_properties, toxicity, antimicrobial_activity, structure_prediction, similarity_analysis, mic_score, mic_original
        except json.JSONDecodeError:
            print("❌ 无法解析 agent 输出 JSON：", final_message)
            return {"evaluation_result": [], "timings": {}}

    def parse_batch_evaluation(self, evaluation_outputs: Dict) -> List[Dict]:
        result = evaluation_outputs
        peptides = []
        for i in range(len(result["peptide_sequence"])):
            peptides.append({
                "sequence": result["peptide_sequence"][i],
                "physicochemical_propertie":result["physicochemical_properties"][i],
                "amp_score": result["antimicrobial_activity"][i],
                "toxicity_score": result["toxicity"][i],
                "plddt_score": result["structure_prediction"][i],
                "similarity_score": result["similarity_analysis"][i],
                "mic_score": result["mic_score"][i],
                "mic_original": result["mic_original"][i]
            })
        return peptides
    
    def extract_overall_score(self, response_text: str) -> float:
        match = re.search(r'\{["\']overall["\']\s*:\s*([0-9.]+)\}', response_text)
        if match:
            score_str = match.group(1)
            try:
                return float(score_str)
            except ValueError:
                raise ValueError(f"解析失败，无法将提取的 '{score_str}' 转为 float")
        else:
            raise ValueError("未找到包含 'overall' 分数的 JSON 格式")

    # def compute_rewards(self, avg_mic_score: float,avg_amp_score:float,llm_overall_score: float) -> float:  
    #     if not self.use_default_reward:                
    #         return compute_rewards(avg_mic_score, avg_amp_score, llm_overall_score)
    #     else:
    #         # def clip01(x: float) -> float:
    #         #     return max(0.0, min(1.0, x))

    #         # def safe_pow(base: float, expv: float, eps: float) -> float:
    #         #     return math.pow(max(base, eps), expv)  # 避免 0**负数 或 0**0

    #         # wa, wb = 0.45, 0.55   # Stage0 偏探索可设 (0.4, 0.6)；Stage1 均衡可调到 (0.5, 0.5)
    #         # eps = 1e-6

    #         # G = safe_pow(avg_mic_score, wa, eps) * safe_pow(avg_amp_score, wb, eps)
    #         # r = G
    #         # return clip01(r)


    #         """
    #         Reward function for AMP generation with three signals Sa, Sb, Sc.

    #         Design goals:
    #         - Emphasize Sa and Sb equally and more than Sc: Sa ≈ Sb > Sc.
    #         - Monotonic, smooth, numerically stable, bounded in [0, 1].
    #         - Encourage 'both-strong' behavior on Sa and Sb via harmonic mean.

    #         Assumptions:
    #         - Inputs are intended to be in [0, 1]. Values outside will be clipped.

    #         Returns:
    #         - A scalar reward in [0, 1].
    #         """

    #         # --- numeric safety & clipping ---
    #         Sa=avg_mic_score
    #         Sb=avg_amp_score
    #         Sc=llm_overall_score

    #         EPS = 1e-8
    #         Sa = 0.0 if Sa is None else Sa
    #         Sb = 0.0 if Sb is None else Sb
    #         Sc = 0.0 if Sc is None else Sc

    #         # Clip to [0,1] for stability and bounded reward
    #         if Sa < 0.0: Sa = 0.0
    #         if Sa > 1.0: Sa = 1.0
    #         if Sb < 0.0: Sb = 0.0
    #         if Sb > 1.0: Sb = 1.0
    #         if Sc < 0.0: Sc = 0.0
    #         if Sc > 1.0: Sc = 1.0

    #         # --- fuse Sa & Sb with harmonic mean (short-board sensitive & monotonic) ---
    #         # H_ab in [0,1]; use EPS to avoid division-by-zero when Sa+Sb=0
    #         H_ab = (2.0 * Sa * Sb) / (Sa + Sb + EPS)

    #         # --- emphasize H_ab over Sc ---
    #         # Weights reflect: Sa ≈ Sb > Sc  → give H_ab higher weight
    #         w_ab = 0.80  # weight on the "A&B" fused objective
    #         w_c  = 0.20  # weight on C (less important but non-negligible)

    #         reward = w_ab * H_ab + w_c * Sc

    #         # reward is already in [0,1] given inputs in [0,1]
    #         # (H_ab ∈ [0,1], Sc ∈ [0,1], convex combination)
    #         return reward
    def compute_rewards(self, avg_mic_score: float, avg_amp_score: float,
                        llm_overall_score: float) -> float:
        # 如果是 RL Scientist 生成的自定义 reward 环境，走原来的 JIT 版函数
        if not self.use_default_reward:
            return compute_rewards(avg_mic_score, avg_amp_score, llm_overall_score)

        # ==== 1. 基础数值预处理 ====
        Sa = 0.0 if avg_mic_score is None else avg_mic_score
        Sb = 0.0 if avg_amp_score is None else avg_amp_score
        Sc = 0.0 if llm_overall_score is None else llm_overall_score

        # Clip 到 [0,1]（按你原来的逻辑）
        Sa = max(0.0, min(1.0, Sa))
        Sb = max(0.0, min(1.0, Sb))
        Sc = max(0.0, min(1.0, Sc))

        EPS = 1e-8

        # ==== 2. 针对 Macrel (Sb) 的消融 ====
        # 实验 3: -Sb
        # 实验 5: -Vb, -Sb
        # 实验 6: -Va, -Sb
        # 实验 7: -Va, -Vb, -Sb
        if self.manual_reward_no_sb:
            # 不用 Macrel (Sb)，只用 Sa 和 Sc 做简单平均
            return 0.5 * Sa + 0.5 * Sc

        # ==== 3. baseline（以及只消 Va/Vb 的实验 1,2,4）保持原 reward ====
        H_ab = (2.0 * Sa * Sb) / (Sa + Sb + EPS)
        w_ab = 0.80
        w_c  = 0.20
        reward = w_ab * H_ab + w_c * Sc
        return reward


            

if __name__ == "__main__":
    seqs = ["RRIRRPRLPRPRVPRPRI"]
    phys_chem_results = json.loads(biopython_protein_analysis(seqs=seqs)).get("biopython_protein_analysis", [])
    print(phys_chem_results)








import math, torch

@torch.jit.script
def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

@torch.jit.script
def to01_from_m11(x: float) -> float:
    return 0.5 * (x + 1.0)

@torch.jit.script
def safe_sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

@torch.jit.script
def safe_pow(base: float, expv: float, eps: float) -> float:
    return math.pow(max(base, eps), expv)

@torch.jit.script
def compute_rewards(Sa: float, Sb: float, Sc: float) -> float:
    # Stage 0: 进一步优化门控以增强稳定性与安全性
    wa, wb = 0.5, 0.5
    eps = 1e-6
    G = safe_pow(Sa, wa, eps) * safe_pow(Sb, wb, eps)

    Sc01 = to01_from_m11(Sc)
    alpha, tau = 6.0, 0.52  # 调整 α 稳定性与控制边界
    g = safe_sigmoid(alpha * (Sc01 - tau))

    r = G * g
    return clip01(r)
