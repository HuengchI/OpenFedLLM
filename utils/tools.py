import os
import json
from tqdm.auto import tqdm
from functools import partial
from dataclasses import asdict

def is_main_process(script_args):
    return script_args.local_rank <= 0


def df_prepend_instruction(df, target_column_name, instruction_str):
    tqdm.pandas()

    def prepend_instruction(row, target_column_name, instruction_str):
        row[target_column_name] = f"{instruction_str}\n{row[target_column_name]}"
        return row

    df = df.progress_apply(partial(prepend_instruction,
                                   target_column_name=target_column_name,
                                   instruction_str=instruction_str),
                           axis=1)

    return df


def dump_args(args, output_dir, file_name="args.json"):
   with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(asdict(args), f, indent=4)
