import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
import openai
import re
import subprocess
from pathlib import Path
import shutil
import time 
from tqdm import trange

from utils.misc import * 
from utils.create_task import create_task
from utils.extract_task_code import *
from utils.cleanup import cleanup_ollama_model
# from reward_fn_discriminator import RewardFunctionDiscriminator
from rl_refinement_module.rl_scientist_reward_designer import *
from rl_refinement_module.input_log_inside import *
from utils.compute_logger import compute_logger


ROOT_DIR = os.getcwd()
amp_generator_ROOT_DIR = f"{ROOT_DIR}/../Generation_Module"
MIC_REGRESSION_ROOT_DIR = f"{ROOT_DIR}/../AMP_regression/predict"
# SKIP_INDEPENDENT_REVIEWER="Perplexity"


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main_workflow(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")

    task = cfg.env.task
    suffix = cfg.suffix
    model = cfg.model
    default_sign=cfg.default_sign
    skip_independent_reviewer = cfg.skip_independent_reviewer
    real_epoch=cfg.real_epoch
    max_length=cfg.max_length
    sand_box_epoch=cfg.sand_box_epoch
    drift_wordlist=cfg.drift_wordlist
    ablation_mode=cfg.ablation_mode
    experiment=cfg.experiment
    logging.info("Default_sign: " + str(default_sign))
    logging.info("real_epoch: " + str(real_epoch))
    logging.info("max_length: " + str(max_length))
    logging.info(f"Using LLM: {model}")
    logging.info(f"drift_wordlist: {drift_wordlist}")
    logging.info(f"ablation_mode: {ablation_mode}")


    api_key="Your API Key"
    openai_client = openai.OpenAI(api_key=api_key)
    ollama_host  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llama_tag    = os.getenv("OLLAMA_LLAMA_TAG", "llama3.1:8b")
    qwen_tag     = os.getenv("OLLAMA_QWEN_TAG",  "qwen2.5:7b-instruct")

    logging.info(f"Experiment: {experiment}")
    logging.info(f"Ollama host: {ollama_host}")
    logging.info(f"Ollama llama tag: {llama_tag}")
    logging.info(f"Ollama qwen tag: {qwen_tag}")

    logging.info("Task: " + task)



    env_name = cfg.env_name.lower()
    print(env_name)
    env_parent = 'test'
    print(env_parent)
    task_file = f'{ROOT_DIR}/envs/{env_parent}/{env_name}.py'
    task_code_string  = file_to_string(task_file)
    output_file = f"{amp_generator_ROOT_DIR}/evaluation_service.py"

    # Loading all text prompts
    prompt_dir = f'{ROOT_DIR}/utils/prompts'
    Reward_Judge_Agent_Prompt = file_to_string(f'{prompt_dir}/Reward_Judge_Agent_Prompt.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')
    task_code_string = task_code_string.replace(task, task+suffix)
    real_train_stage_iteration = cfg.real_train_stage_iteration
    sandbox_iteration = cfg.sandbox_iteration
    reward_sample_num = cfg.reward_sample_num
    batch_size=cfg.batch_size
    sandbox_memory_dir = "outputs/memory_dir"

    # Create Task YAML files
    create_task(amp_generator_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    finetuned_model_path = os.environ.get("FINETUNED_MODEL_PATH1")
    if finetuned_model_path is not None:
        print(f"✅ 检测到环境变量 MODEL_PATH，使用路径: {finetuned_model_path}")
    else:
        finetuned_model_path = cfg.get("model_path", f"{amp_generator_ROOT_DIR}/model_ckpt")
        print(f"⚠️ 未设置环境变量 MODEL_PATH，改为使用 config 默认值: {finetuned_model_path}")
    
    # # Execute the initial AMP Generator
    set_freest_gpu()
    rl_filepath = f"reward_code_train_base.txt"
    warmup_data_path = f"{amp_generator_ROOT_DIR}/data/warmup/warmup_AMP_Escherichia_coli.csv"
    reward_id = f"iter_0_sandbox_iter_0_response_0" 
    print(f'LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python {amp_generator_ROOT_DIR}/run_ppo_training.py --epochs 1 --batch_size {batch_size} --initialize_model --use_default_reward --warmup_dir {warmup_data_path} --reward_id {reward_id} --stage_id 0 --model_path {finetuned_model_path} --save_dir real_model_output_0/ppo_ckpt --amp_generator_root {amp_generator_ROOT_DIR} --amp_regression_root {MIC_REGRESSION_ROOT_DIR} --workspace_dir {workspace_dir} --default_sign {default_sign} --max_length {max_length} --drift_wordlist {drift_wordlist} --ablation_mode {ablation_mode} --experiment {experiment} --ollama_host {ollama_host} --llama_tag {llama_tag} --qwen_tag {qwen_tag} --skip_independent_reviewer {skip_independent_reviewer}')
    
    # warmup start
    compute_logger.start_gpu_block(
        tag=f"warmup_{reward_id}",
        epoch=0,
        stage="warmup_0",
        gpu_count=1,   # 若你知道是多卡，这里改成对应卡数
    )
    with open(rl_filepath, 'w') as f:
        process = subprocess.Popen(
            [
                'bash', '-c',
                f'LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python {amp_generator_ROOT_DIR}/run_ppo_training.py --epochs 1 --batch_size {batch_size} --initialize_model --use_default_reward --warmup_dir {warmup_data_path} --reward_id {reward_id} --stage_id 0 --model_path {finetuned_model_path} --save_dir real_model_output_0/ppo_ckpt --amp_generator_root {amp_generator_ROOT_DIR} --amp_regression_root {MIC_REGRESSION_ROOT_DIR} --workspace_dir {workspace_dir} --default_sign {default_sign} --max_length {max_length} --drift_wordlist {drift_wordlist} --ablation_mode {ablation_mode} --experiment {experiment} --ollama_host {ollama_host} --llama_tag {llama_tag} --qwen_tag {qwen_tag} --skip_independent_reviewer {skip_independent_reviewer}'
            ],
            stdout=f,
            stderr=f
        )
        process.wait()
    compute_logger.end_gpu_block()  # 新增：结束计时
    block_until_training(rl_filepath, log_status=True, iter_num=0, response_id=0)
    process.communicate()

    # generation loop start
    input_log_outside_scientist = InputLog(role="scientist")
    input_log_outside_critic = InputLog(role="critic")
    previous_input_log_path = f"{sandbox_memory_dir}/iter_0/iter_0_sandbox_iter_0_response_0_memory_log.txt"
    load_memory_log_file(previous_input_log_path,input_log_outside_scientist, input_log_outside_critic,default_code_str="None")

    best_input_log_scientist=input_log_outside_scientist
    best_input_log_critic=input_log_outside_critic
    best_rid = "None"
    # for iter in range(1,real_train_stage_iteration+1):
    for iter in trange(1, real_train_stage_iteration+1, desc="Real stages"):        
        # for sandbox_iter in range(1,sandbox_iteration+1):
        for sandbox_iter in trange(1, sandbox_iteration+1, desc="Sandbox iters"):
            total_samples = 0
            responses = []
            response_cur = None
            chunk_size = 1
            logging.info(f"Iteration {iter}; Sandbox Iteration {sandbox_iter}: Generating {reward_sample_num} samples with {cfg.model}")
            compute_logger.set_context(epoch=iter, stage=f"sandbox_{sandbox_iter}_design")            
            while True:
                if total_samples >= cfg.sample:
                    break
                for attempt in range(10):
                    try:
                        response_cur = run_design_loop(best_input_log_scientist.to_str_scientist(), best_input_log_critic.to_str_critic(), ROOT_DIR, Gpt5Client(api_key=api_key), max_rounds=4)
                        break
                    except Exception as e:
                        logging.info(f"Attempt {attempt+1} failed with error: {e}")
                        print(response_cur)
                        time.sleep(3)
                if response_cur is None:
                    logging.info("Current Response is Empty!!")
                    exit()

                if response_cur.approved:
                    total_samples += chunk_size
                    # print(f"For reward {total_samples}------------------------------------------")
                    # print(response_cur.reward_code)
                    responses.append(response_cur.reward_code)

            code_runs = [] 
            rl_runs = []
            code_paths = []
            input_log_inside_scientist_list = [] # InputLog(role="scientist")
            input_log_inside_critic_list = [] # InputLog(role="critic")
            # for response_id in range(reward_sample_num):
            for response_id in trange(reward_sample_num, desc=f"Train rewards (sandbox {sandbox_iter})"):
                if sandbox_iter==1:
                    input_log_inside_scientist = InputLog(role="scientist")
                    input_log_inside_critic = InputLog(role="critic")
                else: 
                    input_log_inside_scientist = copy.deepcopy(best_input_log_scientist)
                    input_log_inside_critic = copy.deepcopy(best_input_log_critic)

                response_cur = responses[response_id]
                reward_id = f"iter_{iter}_sandbox_iter_{sandbox_iter}_response_{response_id}" 
                input_log_file = f"{sandbox_memory_dir}/iter_{iter}/{reward_id}_memory_log.txt"
                logging.info(f"Iteration {iter}; sandbox_iter {sandbox_iter}: Processing Code Run {response_id}")
                # logging.info(response_cur)

                code_string = response_cur
                code_runs.append(code_string)
                with open(output_file, 'w') as file:
                    file.writelines(task_code_string + '\n')
                    file.writelines(code_string + '\n')
                
                with open(f"env_iter{iter}_sandbox_iter_{sandbox_iter}_response{response_id}_rewardonly.py", 'w') as file:
                    file.writelines(code_string + '\n')

                # Copy the generated environment code to hydra output directory for bookkeeping
                reward_code_path = f"iter_{iter}_sandbox_iter_{sandbox_iter}_response_{response_id}.py"
                code_paths.append(reward_code_path)
                shutil.copy(output_file, reward_code_path)

                # Find the freest GPU to run GPU-accelerated RL
                set_freest_gpu()

                # Execute the python file with flags
                rl_filepath = f"env_iter{iter}_sandbox_iter_{sandbox_iter}_response{response_id}.txt"
                
                # 新增：这一条 PPO 训练视作一个 GPU block
                compute_logger.start_gpu_block(
                    tag=reward_id,
                    epoch=iter,
                    stage=f"sandbox_{sandbox_iter}",
                    gpu_count=1,
                )
                with open(rl_filepath, 'w') as f:
                    process = subprocess.Popen(
                        [
                            'bash', '-c',
                            f'LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python {amp_generator_ROOT_DIR}/run_ppo_training.py --epochs {sand_box_epoch} --batch_size {batch_size} --iter {iter} --reward_id {reward_id} --stage_id IN --response_id {response_id} --model_path real_model_output_{iter-1}/ppo_ckpt --amp_generator_root {amp_generator_ROOT_DIR} --amp_regression_root {MIC_REGRESSION_ROOT_DIR} --workspace_dir {workspace_dir} --default_sign {default_sign} --max_length {max_length} --drift_wordlist {drift_wordlist} --ablation_mode {ablation_mode} --experiment {experiment} --ollama_host {ollama_host} --llama_tag {llama_tag} --qwen_tag {qwen_tag} --skip_independent_reviewer {skip_independent_reviewer}'
                        ],
                        stdout=f,
                        stderr=f
                    )
                    process.wait()  # 确保当前 PPO 完全执行并显存释放
                compute_logger.end_gpu_block()  # 新增
                block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
                rl_runs.append(process)
                load_memory_log_file(input_log_file, input_log_inside_scientist, input_log_inside_critic, default_code_str=code_string)
                input_log_inside_scientist_list.append(input_log_inside_scientist)
                input_log_inside_critic_list.append(input_log_inside_critic)
            
            # To select best reward function. Step 2: Pareto Front method
            selector_prompt = f"The following are the performance metrics after PPO training using {len(input_log_inside_scientist_list)} different reward functions：\n\n"
            for input_log_i in range(len(input_log_inside_scientist_list)):
                try:

                    selector_prompt += f'[M{input_log_i+1}]{input_log_inside_scientist_list[input_log_i].to_str_scientist()}\n'
 
                except: 
                    selector_prompt += execution_error_feedback.format(traceback_msg="The evaluation produced no meaningful results, and the generation quality is poor. Please reconsider the design of the reward function!")

            selector_message = [
                {"role": "system", "content": Reward_Judge_Agent_Prompt},
                {"role": "user", "content": selector_prompt}
            ]

            with compute_logger.context(
                epoch=iter,
                stage=f"sandbox_{sandbox_iter}_judge",
                agent="RewardJudgeAgent",
            ):
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=selector_message,
                    temperature=0.3,
                    n=1
                )
                # 利用 logger 自动从 response.usage 中取 token
                compute_logger.log_openai_response(
                    response,
                    api_name=model,
                    epoch=iter,
                    stage=f"sandbox_{sandbox_iter}_judge",
                    agent="RewardJudgeAgent",
                )

            llm_response = response.choices[0].message.content
            logging.info(f"Reward Judge Agent发表了如下评论: {llm_response}")

            match = re.search(r'(?im)^\s*\[M\s*(\d+)\]', llm_response)
            if match:
                best_input_log_idx = int(match.group(1)) -1  # -> 1

            # print(llm_response)
            # print(best_input_log_idx)
            cur_sanbox_iter_best_best_input_log_scientist = input_log_inside_scientist_list[best_input_log_idx]
            cur_sanbox_iter_best_best_input_log_critic = input_log_inside_critic_list[best_input_log_idx]
            if (iter>1 or sandbox_iter>1) and cur_sanbox_iter_best_best_input_log_scientist.is_strictly_worse_than(best_input_log_scientist):
                print(f"Reward Judge Agent作出了选择{cur_sanbox_iter_best_best_input_log_scientist.last_reward_id()}\n最优reward沙盒训练后差于初始reward，使用旧的reward函数{best_input_log_scientist.last_reward_id()}\n停止沙盒循环!!!")
                break
            else:
                print(f"Reward Judge Agent作出了选择{cur_sanbox_iter_best_best_input_log_scientist.last_reward_id()}\n最优reward沙盒训练后优于初始reward，使用新的reward函数{cur_sanbox_iter_best_best_input_log_scientist.last_reward_id()}")
                best_input_log_scientist = cur_sanbox_iter_best_best_input_log_scientist
                best_input_log_critic = cur_sanbox_iter_best_best_input_log_critic
        
        set_freest_gpu()
        best_rid = best_input_log_scientist.last_reward_id()
        selected_code_path=f"{best_rid}.py"
        print(f"由于我们选择了reward函数 {best_rid}，对应的代码在{selected_code_path},我们将其copy到{output_file}")
        print(selected_code_path)
        shutil.copy(selected_code_path, output_file)
        
        # Execute the python file with flags
        rl_filepath = f"reward_code_train_{iter}.txt"
        print(f"下面进行第{iter}轮真实训练:")
        
        print([
            'bash', '-c',
            f'LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python {amp_generator_ROOT_DIR}/run_ppo_training.py --epochs {real_epoch} --batch_size {batch_size} --iter {iter} --reward_id {best_rid} --stage_id {iter} --response_id {response_id} --real_model_train --model_path real_model_output_{iter-1}/ppo_ckpt --save_dir real_model_output_{iter}/ppo_ckpt --amp_generator_root {amp_generator_ROOT_DIR} --amp_regression_root {MIC_REGRESSION_ROOT_DIR} --workspace_dir {workspace_dir} --default_sign {default_sign} --max_length {max_length} --drift_wordlist {drift_wordlist} --ablation_mode {ablation_mode} --experiment {experiment} --ollama_host {ollama_host} --llama_tag {llama_tag} --qwen_tag {qwen_tag} --skip_independent_reviewer {skip_independent_reviewer}'
        ],)

        compute_logger.start_gpu_block(
            tag=f"real_train_{best_rid}",
            epoch=iter,
            stage=f"real_train_{iter}",
            gpu_count=1,
        )
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen(
                [
                    'bash', '-c',
                    f'LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python {amp_generator_ROOT_DIR}/run_ppo_training.py --epochs {real_epoch} --batch_size {batch_size} --iter {iter} --reward_id {best_rid} --stage_id {iter} --response_id {response_id} --real_model_train --model_path real_model_output_{iter-1}/ppo_ckpt --save_dir real_model_output_{iter}/ppo_ckpt --amp_generator_root {amp_generator_ROOT_DIR} --amp_regression_root {MIC_REGRESSION_ROOT_DIR} --workspace_dir {workspace_dir} --default_sign {default_sign} --max_length {max_length} --drift_wordlist {drift_wordlist} --ablation_mode {ablation_mode} --experiment {experiment} --ollama_host {ollama_host} --llama_tag {llama_tag} --qwen_tag {qwen_tag} --skip_independent_reviewer {skip_independent_reviewer}'
                ],
                stdout=f,
                stderr=f
            )
            process.wait()  # 确保当前 PPO 完全执行并显存释放
        compute_logger.end_gpu_block()
        block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
        process.communicate()

        real_model_train_input_log_file = "real_model_" + f"{sandbox_memory_dir}/iter_{iter}/{best_rid}_memory_log.txt"
        load_memory_log_file(real_model_train_input_log_file,input_log_outside_scientist, input_log_outside_critic, best_input_log_scientist.last_code_str())
        best_input_log_scientist=input_log_outside_scientist
        best_input_log_critic=input_log_outside_critic
        # 新增：保存开销统计
    stats_path = Path.cwd() / "compute_stats.json"
    compute_logger.save_json(str(stats_path))
    logging.info(f"[ComputeLogger] 保存开销统计到 {stats_path}")

if __name__ == "__main__":
    main_workflow()