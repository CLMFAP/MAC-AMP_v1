# from evaluation_service import Evaluator
# evaluator = Evaluator(use_default_reward=True)
# evaluation_result, physicochemical_properties, toxicity, antimicrobial_activity, structure_prediction, similarity_analysis, mic_prediction = evaluator.get_evaluation_outputs_from_agent(sequences)

import pandas as pd
import numpy as np
from tqdm import tqdm
from evaluation_service import Evaluator

# 文件路径
input_path = "?????/Your file.csv"
output_path = "?????/Your folder/result.csv"

# 读取数据
df = pd.read_csv(input_path)
evaluator = Evaluator(use_default_reward=True)

# 初始化结果列表
results = []

# 每10行处理一次
batch_size = 10
for i in tqdm(range(0, len(df), batch_size), desc="Evaluating"):
    batch_df = df.iloc[i:i+batch_size]
    sequences = batch_df['sequence'].tolist()
    targets = batch_df['Targets'].tolist()

    # 计算 log10_MIC 和 mic_score
    log10_mics = np.log10(np.array(targets, dtype=np.float32))
    mic_scores = [(4.0 - min(max(m, -2.0), 4.0)) / 6.0 for m in log10_mics]

    try:
        evaluation_result, _, toxicity, antimicrobial_activity, structure_prediction, similarity_analysis, mic_prediction = evaluator.get_evaluation_outputs_from_agent(sequences)

        for j in range(len(sequences)):
            result = {
                "sequence": sequences[j],
                "Targets": targets[j],
                "log10_MIC": log10_mics[j],
                "mic_score": mic_scores[j],
                "toxicity": toxicity[j],
                "antimicrobial_activity": antimicrobial_activity[j],
                "structure_prediction": structure_prediction[j],
                "mic_prediction": mic_prediction[j],
            }
            results.append(result)

    except Exception as e:
        print(f"❌ Error in batch {i}-{i+batch_size}: {e}")
        continue

# 转成 DataFrame 并保存
result_df = pd.DataFrame(results)
result_df.to_csv(output_path, index=False)
print(f"✅ Evaluation results saved to: {output_path}")
