import subprocess
import os
import json
import logging

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()
    
import subprocess
def cleanup_ollama_model(model_name: str):
    try:
        subprocess.run(["ollama", "stop", model_name], check=True)
        print(f"✅ Ollama 模型 {model_name} 已成功停止。")
    except Exception as e:
        print(f"⚠️ Ollama 模型停止失败: {e}")

def get_freest_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if result.returncode != 0:
            print("[ERROR] nvidia-smi failed:", result.stderr.decode('utf-8'))
            return None

        memory_list = [int(x) for x in result.stdout.decode('utf-8').strip().split('\n')]
        freest_gpu = memory_list.index(max(memory_list))
        print(f"[INFO] Freest GPU index: {freest_gpu}, memory free: {memory_list[freest_gpu]} MB")
        return freest_gpu

    except Exception as e:
        print("[ERROR] Failed to get GPU info:", e)
        return None

def set_freest_gpu():
    gpu_id = get_freest_gpu()
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[INFO] Set CUDA_VISIBLE_DEVICES to {gpu_id}")
    else:
        print("[WARNING] No GPU available, CUDA_VISIBLE_DEVICES not set.")



def block_until_training(rl_filepath, log_status=False, iter_num=-1, response_id=-1):
    while True:
        rl_log = file_to_string(rl_filepath)
        # print(rl_log)
        if "fps step: PPO training loop started" in rl_log or "Traceback" in rl_log:
            if log_status and "fps step:" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            if log_status and "Traceback" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break


if __name__ == "__main__":
    print(get_freest_gpu())