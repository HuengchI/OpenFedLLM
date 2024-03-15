"""
e.g.
python predict.py --base_model_path /home/models/Llama-2-7b-hf \
--template alpaca \
--lora_path ../../output/CodeAlpaca-20k_1000_fedavg_c1s1_i1000_b1a1_l8192_r32a64_20240315132536/checkpoint-r1-s1000 \
--test_set_path ../../datasets/DISC-Law-SFT/jud_doc_sum/jud_doc_sum_test_split.jsonl \
--output_dir ./prediction_output \
--max_new_token 2048
"""

import json
import argparse
import os
from tqdm import tqdm
import torch
from datetime import datetime
import pandas as pd

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
UTILS_LIB_PATH = "../../"
sys.path.append(os.path.abspath(UTILS_LIB_PATH))
from utils.conversation import get_conv_template

temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default=None)
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--template", type=str, default="alpaca")
parser.add_argument("--max_new_token", type=int, default=1024)
parser.add_argument("--test_set_path", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)


args = parser.parse_args()

# ============= Extract model name from the path. The name is used for saving results. =============
if args.lora_path:
    pre_str, checkpoint_str = os.path.split(args.lora_path)
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = checkpoint_str.split("-")[-1]
    model_name = f"{exp_name}_{checkpoint_id}"
else:
    pre_str, last_str = os.path.split(args.base_model_path)
    if last_str.startswith("full"):                 # if the model is merged as full model
        _, exp_name = os.path.split(pre_str)
        checkpoint_id = last_str.split("-")[-1]
        model_name = f"{exp_name}_{checkpoint_id}"
    else:
        model_name = last_str                       # mainly for base model


# ============= Load model and tokenizer =============
model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16).to('cuda')    # float16 to run inference of 7B model on 3090 GPU
if args.lora_path:
    model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

# ============= Load dataset =============
test_ds = pd.read_json(args.test_set_path, lines=True)

# ============= Generate answers =============
print(f">> The template is:\n{get_conv_template(args.template).system_message}")
output_file_path = os.path.join(args.output_dir, f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jsonl")
print(f">> Outputs are saving to {output_file_path}")

pbar = tqdm(total=len(test_ds))
for _, row in tqdm(test_ds.iterrows()):

    temperature = 0.7

    conv = get_conv_template(args.template)

    conv.append_message(conv.roles[0], row['input'])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer([prompt]).input_ids

    output_ids = model.generate(
        input_ids=torch.as_tensor(input_ids).cuda(),
        do_sample=False,
        temperature=temperature,
        max_new_tokens=args.max_new_token,
    )
    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]

    # be consistent with the template's stop_token_ids
    if conv.stop_token_ids:
        stop_token_ids_index = [
            i
            for i, id in enumerate(output_ids)
            if id in conv.stop_token_ids
        ]
        if len(stop_token_ids_index) > 0:
            output_ids = output_ids[: stop_token_ids_index[0]]

    output = tokenizer.decode(
        output_ids,
        spaces_between_special_tokens=False,
    )

    if conv.stop_str and output.find(conv.stop_str) > 0:
        output = output[: output.find(conv.stop_str)]
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for special_tok in special_token:
                output = output.replace(special_tok, "")
        else:
            output = output.replace(special_token, "")
    output = output.strip()

    # Dump answers
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_file_path, "a") as fout:
        ans_json = {
            "id": row['id'],
            "output": output,
        }
        fout.write(json.dumps(ans_json) + "\n")

    # display output
    print(output)

    pbar.update()
pbar.close()
