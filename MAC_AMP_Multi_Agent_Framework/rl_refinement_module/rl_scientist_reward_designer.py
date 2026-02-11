"""
RL Scientist Agent & Critical Agent — reward function design loop (max 4 rounds).

Update:
- Removed the `if self.use_stub:` stub logic.
- Only real GPT-5 API call logic remains.
- Prompts are loaded from `prompt_scientist.txt` and `prompt_critic.txt`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json
import re
import os
import textwrap
from openai import OpenAI
from .reward_fn_discriminator import validate_reward_code
from utils.compute_logger import compute_logger  # 新增：记录 API 调用与 token 开销

api_key="Put your key here"
# -----------------------------------------------------------------------------
# 1) Utility to load prompt from file
# -----------------------------------------------------------------------------
def load_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return f"[Prompt file {path} not found]"

# -----------------------------------------------------------------------------
# 2) gpt-5 client wrapper (real only)
# -----------------------------------------------------------------------------
@dataclass
class Gpt5Client:
    api_key: Optional[str] = None

    def chat(
        self,
        system: str,
        user: str,
        *,
        agent: Optional[str] = None,
        stage: Optional[str] = None,
        epoch: Optional[int] = None,
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> str:
        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system}, {"role":"user","content":user}]
        )

        compute_logger.log_openai_response(
            resp,
            api_name=model,
            epoch=epoch,
            stage=stage,
            agent=agent,
        )
        return resp.choices[0].message.content

# -----------------------------------------------------------------------------
# 3) Dialogue memory to persist messages across rounds
# -----------------------------------------------------------------------------
@dataclass
class Turn:
    round_idx: int
    scientist_proposal: Optional[str] = None
    critic_feedback: Optional[str] = None

class DialogueMemory:
    def __init__(self) -> None:
        self.turns: List[Turn] = []

    def start_round(self, idx: int) -> None:
        self.turns.append(Turn(round_idx=idx))

    def record_scientist(self, idx: int, proposal: str) -> None:
        self.turns[-1].scientist_proposal = proposal

    def record_critic(self, idx: int, feedback: str) -> None:
        self.turns[-1].critic_feedback = feedback

    def render_for_scientist(self) -> str:
        lines = []
        for t in self.turns:
            if t.scientist_proposal:
                lines.append(f"[Round {t.round_idx}] Scientist Proposal\n{t.scientist_proposal}")
            if t.critic_feedback:
                lines.append(f"[Round {t.round_idx}] Critic Feedback\n{t.critic_feedback}")
        return "\n\n".join(lines)

    def render_for_critic(self) -> str:
        lines = []
        for t in self.turns:
            if t.scientist_proposal:
                lines.append(f"[Round {t.round_idx}] Proposed Reward Function\n{t.scientist_proposal}")
            if t.critic_feedback:
                lines.append(f"[Round {t.round_idx}] Prior Critique\n{t.critic_feedback}")
        return "\n\n".join(lines)

# -----------------------------------------------------------------------------
# 4) Agents
# -----------------------------------------------------------------------------
class RLScientistAgent:
    def __init__(self, llm: Gpt5Client, main_workflow_root, prompt_file: str = "rl_refinement_module/prompt_scientist.txt"):
        prompt_file = os.path.join(main_workflow_root, prompt_file)
        self.llm = llm
        self.system_prompt = load_prompt(prompt_file)

    def propose_reward(self, sci_text: str, memory_text: str) -> str:
        user = memory_text
        self.system_prompt = self.system_prompt.replace("{Input_Log_RL_Scientist_Agent}", sci_text)
        # print("$"*80)
        # print(self.system_prompt)
        # print("$"*80)
        # print(user)
        # print("$"*80)
        # code = self.llm.chat(self.system_prompt, user)

        with compute_logger.context(agent="RLScientistAgent", stage="reward_design_scientist"):
            code = self.llm.chat(
                self.system_prompt,
                user,
                agent="RLScientistAgent",
                stage="reward_design_scientist",
            )
        return code.strip()

class CriticalAgent:
    def __init__(self, llm: Gpt5Client, main_workflow_root, prompt_file: str = "rl_refinement_module/prompt_critic.txt"):
        prompt_file = os.path.join(main_workflow_root, prompt_file)
        self.llm = llm
        self.system_prompt = load_prompt(prompt_file)

    def evaluate(self, critic_text: str, memory_text: str) -> Dict[str, Any]:
        user = memory_text
        self.system_prompt = self.system_prompt.replace("{Input_Log_Critical_Agent}", critic_text)
        # print("$"*80)
        # print(self.system_prompt)
        # print("$"*80)
        # print(user)
        # print("$"*80)
        # resp = self.llm.chat(self.system_prompt, user)
        with compute_logger.context(agent="CriticalAgent", stage="reward_design_critic"):
            resp = self.llm.chat(
                self.system_prompt,
                user,
                agent="CriticalAgent",
                stage="reward_design_critic",
            )
        return resp.strip()

# -----------------------------------------------------------------------------
# 5) Orchestrator — up to 4 rounds, early stop on approval
# -----------------------------------------------------------------------------
@dataclass
class LoopResult:
    approved: bool
    reward_code: Optional[str]
    critic_reason: str
    rounds_done: int
    final_payload: Optional[Dict[str, Any]]

import re

def extract_reward_block(text: str) -> str:
    """
    优先提取 [Reward Function:] 后的最后一个 ```python 代码块；
    若缺失该标签，则退化为提取全文最后一个 ``` 代码块；
    再否则直接返回原文（给上游兜底处理）。
    """
    m = re.search(r"\[Reward Function:\]\s*```(?:\w+)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # fallback: 最后一个 fenced code block
    blocks = list(re.finditer(r"```(?:\w+)?\s*([\s\S]*?)```", text))
    if blocks:
        return blocks[-1].group(1).strip()

    return text.strip()

def run_design_loop(
    sci_text: str,
    critic_text: str,
    main_workflow_root:str,
    llm_client: Optional[Gpt5Client] = None,
    max_rounds: int = 4,
) -> LoopResult:
    llm = llm_client or Gpt5Client()
    scientist = RLScientistAgent(llm, main_workflow_root)
    critic = CriticalAgent(llm, main_workflow_root)
    memory = DialogueMemory()

    reward_code: Optional[str] = None
    for epoch in range(1, max_rounds + 1):
        compute_logger.set_context(epoch=epoch)
        memory.start_round(epoch)
        reward_code = scientist.propose_reward(sci_text, memory.render_for_scientist())
        memory.record_scientist(epoch, reward_code)
        # print(f"\n[Round {epoch}] RL Scientist Agent 提案:\n{reward_code}")

        eval_result = critic.evaluate(critic_text, memory.render_for_critic())
        memory.record_critic(epoch, eval_result)
        # print(f"[Round {epoch}] Critical Agent 评价:\n{eval_result}")

        ok = eval_result.strip() == "[Pass:] True\n[Comments:] None\n[/]"
        if ok:
            reward_code_only = extract_reward_block(reward_code)
            reward_code_only = reward_code_only.lstrip("\ufeff")                      # 去 BOM（有些编辑器会带）
            reward_code_only = textwrap.dedent(reward_code_only).lstrip("\n\r \t")    # 去整体缩进 + 去前导空行
            res = validate_reward_code(reward_code_only)
            if res.ok:
                # print("✅ 运行期检查通过")
                return LoopResult(True, reward_code_only, eval_result, max_rounds, None)
            else:
                # print("[AUTO-VALIDATOR FAILED] 不选用这个reward函数，因为最终出圈前复检失败，请修复：\n" + res.msg)
                continue
    return LoopResult(False, reward_code, eval_result, max_rounds, None)

# -----------------------------------------------------------------------------
# 6) Demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sci_text = "[Reward Function:None][Code: None][Epoch 0][Raw MIC Value:0.5, MIC Score:0.67][AMP Score:0.55][Meta Review Score:0.3][/ ]"
    critic_text = "[Reward Function:None][Code: None][Epoch 0][Meta Review Message:Initial comments][/ ]"

    result = run_design_loop(sci_text, critic_text, Gpt5Client(api_key=api_key), max_rounds=4)

    print("Approved:", result.approved)
    print("Rounds:", result.rounds_done)
    print("Critic:", result.critic_reason)
    if result.reward_code:
        print("\n=== Reward Function ===\n")
        print(result.reward_code)
    if result.final_payload:
        print("\n=== Final Payload ===\n")
        print(json.dumps(result.final_payload, ensure_ascii=False, indent=2))