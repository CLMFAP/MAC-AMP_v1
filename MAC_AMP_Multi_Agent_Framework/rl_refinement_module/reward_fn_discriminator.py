# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Callable, Any, Tuple
import sys, types
import textwrap

@dataclass
class ValidationResult:
    ok: bool
    msg: str
    value: Optional[float] = None  # 运行得到的浮点值（若成功）
    fn: Optional[Callable] = None  # 提取到的 compute_rewards（若成功）

def validate_reward_code(code: str, sample: Tuple[float,float,float]=(0.55, 0.60, 0.0)) -> ValidationResult:
    """
    极简验证：直接 exec 代码，拿 compute_rewards 跑一次。
    - 若运行异常：返回 RuntimeError 信息
    - 若输出不可被 float()：返回 BadReturnType
    - 否则 OK，并给出 float 值
    """
    if not isinstance(code, str) or not code.strip():
        return ValidationResult(False, "EmptyInput")

    # --- 注入最小 torch.jit.script stub（避免未安装 torch 的环境报错） ---
    torch_prev = sys.modules.get("torch")
    try:
        torch_stub = types.ModuleType("torch")

        class _Jit:
            def script(self, f=None, *args, **kwargs):
                # 兼容 @torch.jit.script 和 torch.jit.script(fn)
                if f is None:
                    def deco(fn): return fn
                    return deco
                return f

        torch_stub.jit = _Jit()
        sys.modules["torch"] = torch_stub

        g = {}
        try:
            exec(code, g, g)
        except Exception as e:
            return ValidationResult(False, f"ExecError: {e}")

        fn = g.get("compute_rewards")
        if not callable(fn):
            return ValidationResult(False, "MissingFunction: compute_rewards")

        Sa, Sb, Sc = sample
        try:
            out = fn(Sa, Sb, Sc)
        except Exception as e:
            return ValidationResult(False, f"RuntimeError: {e}")

        try:
            val = float(out)
        except Exception:
            return ValidationResult(False, f"BadReturnType: {type(out).__name__}")

        return ValidationResult(True, "OK", val, fn)

    finally:
        # 还原 sys.modules['torch']
        if torch_prev is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = torch_prev


# ----- 简单自测（可选） -----
if __name__ == "__main__":
    code = """
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
        wa, wb = 0.4, 0.6
        eps = 1e-6
        G = safe_pow(Sa, wa, eps) * safe_pow(Sb, wb, eps)
        Sc01 = to01_from_m11(Sc)
        alpha, tau = 5.5, 0.5
        g = safe_sigmoid(alpha * (Sc01 - tau))
        return clip01(G * g)
"""
    code = code.lstrip("\ufeff")                      # 去 BOM（有些编辑器会带）
    code = textwrap.dedent(code).lstrip("\n\r \t")    # 去整体缩进 + 去前导空行
    res = validate_reward_code(code)
    print(res.ok, res.msg, res.value)

