do_score = True
cuda_devices: str = "1"
template_spec = "RRS.Trivial"
train_set_source_column = "findings, background"
train_set_target_column = "impression"
dataset_id_column_specs = "study_id, subject_id"

#######################################################################

import os
import subprocess
import random
from datetime import datetime
from utils.random_run_name import generate_run_name

# ====================== Starting Args Definition ======================

model_name_or_path = os.path.abspath("/home/models/OLMo-1B/")

train_set = "//data/huengchi/Datasets/MEDIQA2021/Task3/train.json"
test_set = "//data/huengchi/Datasets/MEDIQA2021/Task3/dev.json"

max_train_steps = str(5000)
train_set_sample = str(5000)
test_set_sample = str(200000)

training_batch_size = 16
training_batch_size_per_GPU = 16
pred_batch_size = 16

output_dir_base = os.path.abspath("./outputs")
exp_name = "RRS"

sft_learning_rate = str(5e-4)

prediction_max_new_tokens = str(500)

# ====================== Finished Args Definition ======================
assert training_batch_size%training_batch_size_per_GPU==0
training__gradient_accu_steps = training_batch_size//training_batch_size_per_GPU

run_name = generate_run_name()

output_dir_name = f"""{exp_name}_{(datetime.now()).strftime('%Y%m%d%H%M%S')}_{run_name}"""
output_dir = os.path.join(output_dir_base, output_dir_name)
output_dir = os.path.abspath(output_dir)


sft_cmd = [
    "//data/huengchi/.bin/anaconda3/envs/torch-2.1-py-3.10/bin/deepspeed",
     "--master_port", str(random.randint(10000, 65535)),
    f"--include=localhost:{cuda_devices}",
    "olmo_sft.py",
    "--learning_rate", sft_learning_rate,
    "--model_name_or_path", model_name_or_path,
    "--custom_local_dataset", train_set,
    "--template", template_spec,
    "--train_set_source_column", train_set_source_column,
    "--train_set_target_column", train_set_target_column,
    "--dataset_sample", train_set_sample,
    "--max_steps", max_train_steps,
    "--batch_size", str(training_batch_size_per_GPU),
    "--gradient_accumulation_steps", str(training__gradient_accu_steps),
    "--warmup_ratio", "0.03",
    "--seq_length", "8192",
    "--peft_lora_r", "32",
    "--peft_lora_alpha", "32",
    "--use_peft",
    "--output_dir", output_dir,
    "--deepspeed", "ds_zero_stage0.json",
    "--bf16", "true",
    "--log_with", "wandb",
    "--logging_steps", "1",
    "--run_name", run_name,
]

lora_path = os.path.join(output_dir, 'checkpoint-final')

pred_cmd = [
    "//data/huengchi/.bin/anaconda3/envs/torch-2.1-py-3.10/bin/deepspeed",
    f"--include=localhost:{cuda_devices}",
    "evaluation/clinical/predict.py",
    "--base_model_path", model_name_or_path,
    "--template", template_spec,
    "--lora_path", lora_path,
    "--test_set_path", test_set,
    "--dataset_id_column_specs", dataset_id_column_specs,
    "--data_sample", test_set_sample,
    "--output_dir", output_dir,
    "--max_new_token", prediction_max_new_tokens,
    "--pred_batch_size", str(pred_batch_size),
]

pred_set = os.path.join(output_dir, 'evals', 'predictions.jsonl')
pred_set = os.path.abspath(pred_set)

score_cmd = [
    "//data/huengchi/.bin/anaconda3/envs/torch-2.1-py-3.10/bin/deepspeed",
    f"--include=localhost:{cuda_devices}",
    "evaluation/clinical/score.py",
    "--prediction_set_path", pred_set,
    "--test_set_path", test_set,
]

env = {
    "http_proxy": "http://10.249.42.241:41122",
    "https_proxy": "http://10.249.42.241:41122",
}

def print_arg_list(arg_list):
    print('[')
    for arg in arg_list[1:]:
        print(f'    "{arg}",')
    print('],')

if __name__ == "__main__":

    process = subprocess.Popen(sft_cmd, env=env)
    process.wait()
    assert process.returncode == 0, f"sft process.returncode={process.returncode}"
    print('> ============= sft Finished =============')

    process = subprocess.Popen(pred_cmd, env=env)
    process.wait()
    assert process.returncode == 0, f"prediction process.returncode={process.returncode}"
    print('> ============= prediction Finished =============')

    if do_score:
        process = subprocess.Popen(score_cmd, env=env)
        process.wait()
        assert process.returncode == 0, f"score process.returncode={process.returncode}"
        print('> ============= score Finished =============')

    print('> ============= All Done! =============')
