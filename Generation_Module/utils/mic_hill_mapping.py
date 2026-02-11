import numpy as np
import pandas as pd

# ------------------------------
# 基础映射：M0=1, α=1
# s = 1 / (1 + (MIC / M0)**alpha) = 1 / (1 + MIC)
# ------------------------------

def hill_score_m0_1_a1_from_mic(mic):
    """
    输入: mic (标量/list/ndarray/Series), 要求 mic > 0
    输出: 与 mic 同长度的 ndarray，范围 (0,1]
    """
    mic_arr = np.asarray(mic, dtype=float)
    s = 1.0 / (1.0 + mic_arr)
    # 非正或 NaN 的 MIC 置为 NaN 分数
    s = np.where(mic_arr > 0, s, 0.0)
    return s

def hill_score_from_mic(mic: float) -> float:
    """
    根据公式 g(x) = 0.75 - 0.25 * tanh(1.46e-6 * (x - 128)^3)
    计算给定 MIC 值的 hill score。

    参数
    ----
    mic : float
        MIC 数值（作为公式中的 x）

    返回
    ----
    float
        对应的 g(x) 值
    """
    return 0.75 - 0.25 * np.tanh(1.46e-6 * (mic - 128) ** 3)

def hill_score_m0_1_a1_from_logmic(log_mic,default_sign):
    """
    输入: log_mic (log10(MIC))，标量/list/ndarray/Series
    输出: 分数 ndarray
    """
    log_arr = np.asarray(log_mic, dtype=float)
    mic = np.power(10.0, log_arr)
    return hill_score_m0_1_a1_from_mic(mic)
    # if default_sign==-1:
    #     return hill_score_m0_1_a1_from_mic(mic)
    # else:
    #     return hill_score_from_mic(mic)

def mic_from_score_m0_1_a1(score):
    """
    反解: 已知分数 s，求 MIC (s ∈ (0,1])
      s = 1 / (1 + MIC)  =>  MIC = 1/s - 1
    """
    s = np.asarray(score, dtype=float)
    return (1.0 / s) - 1.0

# =========================
# Tests
# =========================
if __name__ == "__main__":
    print("Running tests for Hill score (M0=1, α=1): s = 1 / (1 + MIC)\n")

    # ---- Test 1: 标量用例 ----
    # s1 = hill_score_m0_1_a1_from_mic(1.0)[0]
    # print(f"MIC=1.0 -> score={s1:.6f} (expect 0.5)")
    # assert np.isclose(s1, 0.5, atol=1e-12)

    # s2 = hill_score_m0_1_a1_from_logmic(0.0)[0]  # log10(1)=0
    # print(f"logMIC=0.0 -> score={s2:.6f} (expect 0.5)")
    # assert np.isclose(s2, 0.5, atol=1e-12)

    # # ---- Test 2: 列表/混合类型（含负数、None、可解析字符串、非法字符串）----
    # mic_list = [0.1, 1, 10, -5, None, "3", "oops"]
    # scores_mic = hill_score_m0_1_a1_from_mic(mic_list)
    # print("\nInput MIC list:", mic_list)
    # print("Scores from MIC :", np.array2string(scores_mic, precision=6, separator=", "))

    # # 期望: [1/(1+0.1)=0.909090..., 0.5, 1/11≈0.090909, NaN, NaN, 1/4=0.25, NaN]
    # exp = np.array([1/1.1, 0.5, 1/11.0, np.nan, np.nan, 0.25, np.nan], dtype=float)
    # assert np.allclose(scores_mic[[0,1,2,5]], exp[[0,1,2,5]], atol=1e-12)
    # assert np.isnan(scores_mic[3]) and np.isnan(scores_mic[4]) and np.isnan(scores_mic[6])

    # ---- Test 3: 从 logMIC 列表计算（含字符串/非法）----
    log_list = [-1, 0, 0.30103]  # log10(2)≈0.30103
    scores_log = hill_score_m0_1_a1_from_logmic(log_list)
    print("\nInput logMIC list:", log_list)
    print("Scores from logMIC:", np.array2string(scores_log, precision=6, separator=", "))

    # 校验若干点:
    # logMIC=-1 -> MIC=0.1 -> s≈0.9090909
    assert np.isclose(scores_log[0], 1.0/1.1, atol=1e-6)
    # logMIC=0 -> MIC=1 -> s=0.5
    assert np.isclose(scores_log[1], 0.5, atol=1e-12)
    # logMIC≈0.30103 -> MIC≈2 -> s≈1/3
    assert np.isclose(scores_log[2], 1.0/3.0, atol=1e-4)
    # "2" -> MIC=100 -> s=1/101
    assert np.isclose(scores_log[3], 1.0/101.0, atol=1e-12)
    # 非法/None -> NaN
    assert np.isnan(scores_log[4]) and np.isnan(scores_log[5])

    # ---- Test 4: Round-trip 一致性（随机 MIC -> logMIC -> 分数一致）----
    # rng = np.random.default_rng(42)
    # mic_rand = np.clip(rng.lognormal(mean=0.0, sigma=1.0, size=8), 1e-6, None)  # 正 MIC
    # log_rand = np.log10(mic_rand)
    # s_from_mic = hill_score_m0_1_a1_from_mic(mic_rand)
    # s_from_log = hill_score_m0_1_a1_from_logmic(log_rand)
    # print("\nRandom MIC:", np.array2string(mic_rand, precision=6, separator=", "))
    # print("Round-trip |s_mic - s_log| max diff =", float(np.nanmax(np.abs(s_from_mic - s_from_log))))
    # assert np.allclose(s_from_mic, s_from_log, atol=1e-12)

    # # ---- Test 5: 边界/极值稳定性 ----
    # tiny = hill_score_m0_1_a1_from_mic([1e-12])[0]     # 很小 MIC -> 分数接近 1
    # huge = hill_score_m0_1_a1_from_mic([1e12])[0]      # 很大 MIC -> 分数接近 0
    # print(f"\nMIC=1e-12 -> score≈{tiny:.6f} (→1)")
    # print(f"MIC=1e12  -> score≈{huge:.6e} (→0)")
    # assert tiny > 0.999999999999
    # assert huge < 1e-12

    print("\n✅ All tests passed.")