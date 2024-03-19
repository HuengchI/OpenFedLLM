import os
import subprocess
from datetime import datetime

model_name_or_path = os.path.abspath("/home/models/OLMo-1B/")
train_set = os.path.abspath("datasets/mimic-iii-notes/NOTES_len6K_src5900_cov0.5_w_ESGs_train_split.jsonl")
test_set = os.path.abspath("datasets/mimic-iii-notes/NOTES_len6K_src5900_cov0.5_w_ESGs_test_split.jsonl")
template_spec = "SPEER.SPEER"

output_dir_base = os.path.abspath("./outputs")
run_name = "OLMO-1B-Notes"
output_dir_name = f"""{run_name}_{(datetime.now()).strftime('%Y%m%d%H%M%S')}"""
output_dir = os.path.join(output_dir_base, output_dir_name)
output_dir = os.path.abspath(output_dir)

sft_cmd = [
    "//data/huengchi/.bin/anaconda3/envs/torch-2.1-py-3.10/bin/deepspeed",
    "--master_port", "54825",
    "--include=localhost:2",
    "olmo_sft.py",
    "--learning_rate", "5e-6",
    "--model_name_or_path", model_name_or_path,
    "--custom_local_dataset", os.path.abspath(train_set),
    "--template", template_spec,
    "--dataset_sample", "1000",
    "--max_steps", "1000",
    "--batch_size", "1",
    "--gradient_accumulation_steps", "16",
    "--seq_length", "8192",
    "--peft_lora_r", "32",
    "--peft_lora_alpha", "64",
    "--use_peft",
    "--output_dir", output_dir,
    "--deepspeed", "ds_zero_stage0.json",
    "--bf16", "true",
]

lora_path = os.path.join(output_dir, 'checkpoint-final')

pred_cmd = [
    "//data/huengchi/.bin/anaconda3/envs/torch-2.1-py-3.10/bin/deepspeed",
    "--include=localhost:2",
    "evaluation/clinical/predict.py",
    "--base_model_path", model_name_or_path,
    "--template", template_spec,
    "--lora_path", lora_path,
    "--test_set_path", os.path.abspath(test_set),
    "--output_dir", output_dir,
    "--max_new_token", "1800",
]

pred_set = os.path.join(output_dir, 'predictions.jsonl')
pred_set = os.path.abspath(pred_set)

score_cmd = [
    "//data/huengchi/.bin/anaconda3/envs/torch-2.1-py-3.10/bin/deepspeed",
    "--include=localhost:2",
    "evaluation/clinical/score.py",
    "--prediction_set_path", os.path.abspath(pred_set),
    "--test_set_path", os.path.abspath(test_set),
]

env = {
    "http_proxy": "http://10.249.42.241:41122",
    "https_proxy": "http://10.249.42.241:41122",
}

def print_arg_list(arg_list):
    print('[')
    for arg in arg_list:
        print(f'"{arg}",')
    print(']')

if __name__ == "__main__":

    process = subprocess.Popen(sft_cmd, env=env)
    process.wait()

    process = subprocess.Popen(pred_cmd, env=env)
    process.wait()

    process = subprocess.Popen(score_cmd, env)
    process.wait()

    print('Done!')