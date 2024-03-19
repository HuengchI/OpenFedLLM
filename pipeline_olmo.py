import os
import subprocess
from datetime import datetime

do_score = True

# ====================== Starting Args Definition ======================

model_name_or_path = os.path.abspath("/home/models/OLMo-1B/")
train_set = os.path.abspath("datasets/mimic-iii-notes/NOTES_len6K_src5900_cov0.5_w_ESGs_train_split.jsonl")
test_set = os.path.abspath("datasets/mimic-iii-notes/NOTES_len6K_src5900_cov0.5_w_ESGs_test_split.jsonl")
template_spec = "SPEER.SPEER"

max_train_steps = str(1000)
test_set_sample = str(200000)

output_dir_base = os.path.abspath("./outputs")
run_name = "OLMO-1B-Notes"

sft_learning_rate = str(1e-4)

prediction_max_new_tokens = str(300)

# ====================== Finished Args Definition ======================

output_dir_name = f"""{run_name}_{(datetime.now()).strftime('%Y%m%d%H%M%S')}"""
output_dir = os.path.join(output_dir_base, output_dir_name)
output_dir = os.path.abspath(output_dir)

sft_cmd = [
    "//data/huengchi/.bin/anaconda3/envs/torch-2.1-py-3.10/bin/deepspeed",
    "--master_port", "54825",
    "--include=localhost:2",
    "olmo_sft.py",
    "--learning_rate", sft_learning_rate,
    "--model_name_or_path", model_name_or_path,
    "--custom_local_dataset", os.path.abspath(train_set),
    "--template", template_spec,
    "--max_steps", max_train_steps,
    "--batch_size", "1",
    "--gradient_accumulation_steps", "16",
    "--warmup_ratio", "0.03",
    "--seq_length", "8192",
    "--peft_lora_r", "32",
    "--peft_lora_alpha", "64",
    "--use_peft",
    "--output_dir", output_dir,
    "--deepspeed", "ds_zero_stage0.json",
    "--bf16", "true",
    "--log_with", "wandb",
    "--logging_steps", "1",
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
    "--data_sample", test_set_sample,
    "--output_dir", output_dir,
    "--max_new_token", prediction_max_new_tokens,
]

pred_set = os.path.join(output_dir, 'evals', 'predictions.jsonl')
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
    for arg in arg_list[1:]:
        print(f'"{arg}",')
    print('],')

if __name__ == "__main__":

    process = subprocess.Popen(sft_cmd, env=env)
    process.wait()
    assert process.returncode == 0, process.returncode
    print('> ============= sft Finished =============')

    process = subprocess.Popen(pred_cmd, env=env)
    process.wait()
    assert process.returncode == 0, process.returncode
    print('> ============= prediction Finished =============')

    if do_score:
        process = subprocess.Popen(score_cmd, env=env)
        process.wait()
        assert process.returncode == 0, process.returncode
        print('> ============= score Finished =============')

    print('All Done!')