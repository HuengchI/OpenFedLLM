import json
import argparse
import os
from typing import Optional
from tqdm import tqdm
import torch
from datetime import datetime
import pandas as pd
from functools import partial
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
UTILS_LIB_PATH = "."
sys.path.append(os.path.abspath(UTILS_LIB_PATH))
from utils.template import build_generation_prompt
from utils import dump_args
from utils.utils import make_id


@dataclass
class ScriptArguments:
    max_new_token: Optional[int] = field()
    pred_batch_size: Optional[int] = field()
    data_sample: Optional[int] = field(default=200000)
    base_model_path: Optional[str] = field(default=None)
    lora_path: Optional[str] = field(default=None)
    template: Optional[str] = field(default=None)
    test_set_path: Optional[str] = field(default=None)
    dataset_id_column_specs: Optional[str] = field(default='(unset)')
    output_dir: Optional[str] = field(default=None)
    local_rank: Optional[int] = field(default=-1)

parser = argparse.ArgumentParser()
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]


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
model = AutoModelForCausalLM.from_pretrained(args.base_model_path,
                                             trust_remote_code = True,
                                             torch_dtype=torch.float16).to('cuda')    # float16 to run inference of 7B model on 3090 GPU
if args.lora_path:
    model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, padding_side = 'left')

# ============= Load dataset =============
test_ds = pd.read_json(args.test_set_path, dtype=False)
test_ds = test_ds[:args.data_sample]

print(f'> ============ Test set has size {test_ds.shape[0]} ============')

tqdm.pandas()
test_ds = test_ds.progress_apply(partial(build_generation_prompt,
                                         template_spec=args.template,
                                         template_findings='findings',
                                         template_background='background'),
                                 axis=1)

# ============= Dump args =============
args.output_dir = os.path.join(args.output_dir, 'evals')
print(f">> Outputs are saving to {args.output_dir}")
os.makedirs(args.output_dir, exist_ok=True)
dump_args(args, args.output_dir, "prediction_args.json")

# ============= Generate answers =============
output_file_path = os.path.join(args.output_dir, "predictions.jsonl")
print(f">> The template is:\n{args.template}")

pbar = tqdm(total=len(test_ds))

for i in range(0, len(test_ds), args.pred_batch_size):
    batch = test_ds.iloc[i:i+args.pred_batch_size]

    encodes = tokenizer(batch['prompt'].tolist(), padding='longest')
    encodes.pop('token_type_ids')

    output_ids = model.generate(
        **{k:torch.as_tensor(v).cuda() for k,v in encodes.items()},
        do_sample=False,
        max_new_tokens=args.max_new_token,
    )
    output_ids = output_ids.tolist()
    for i in range(args.pred_batch_size):
        output_ids[i] = output_ids[i][len(encodes['input_ids'][i]) :]

    output = tokenizer.batch_decode(
        output_ids,
        spaces_between_special_tokens=False,
    )

    for i in range(args.pred_batch_size):
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output[i] = output[i].replace(special_tok, "")
            else:
                output[i] = output[i].replace(special_token, "")
        output[i] = output[i].strip()

    # Dump answers
    with open(output_file_path, "a") as fout:
        for i in range(args.pred_batch_size):
            row = batch.iloc[i]
            ans_json = {
                "id": make_id(args.dataset_id_column_specs ,row),
                "output": output[i],
            }
            fout.write(json.dumps(ans_json) + "\n")

    # display output
    print(output[0])

    pbar.update(args.pred_batch_size)
pbar.close()
