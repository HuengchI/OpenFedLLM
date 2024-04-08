import os
from typing import Optional
from functools import partial
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass, field
from transformers import HfArgumentParser

import sys
THRID_PARTY_LIB_PATH = (
    ".",
    # "../../SPEER/", # esg_tools
)
for path in THRID_PARTY_LIB_PATH:
    path = os.path.abspath(path)
    if path not in sys.path:
        sys.path.append(path)

from utils.metrics_predefined import (
    RougeMetric,
    BleuMetric,
    ZHBertScoreMetric,
    FactKBMetric,
)
from utils import dump_args
from utils.utils import make_df_id_column, parse_spec


@dataclass
class ScriptArguments:
    prediction_set_path: Optional[str] = field()
    test_set_path: Optional[str] = field()
    local_rank: Optional[int] = field(default=-1)
    metric_src_field_specs: Optional[str] = field(default='findings, background')
    metric_tgt_field_specs: Optional[str] = field(default='impression')
    dataset_id_column_specs: Optional[str] = field(default='study_id, subject_id')

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]


# ============= Load Datasets =============
prediction_df = pd.read_json(args.prediction_set_path, lines=True, dtype=False)

test_df = pd.read_json(args.test_set_path, dtype=False)
test_df = test_df[
    list(set(parse_spec(args.metric_src_field_specs)+parse_spec(args.metric_tgt_field_specs)+parse_spec(args.dataset_id_column_specs)))
    ]

test_df = test_df.apply(partial(make_df_id_column, id_spec=args.dataset_id_column_specs),
                        axis=1)

# ============= Process Datasets =============
result_df = pd.merge(prediction_df, test_df, on='id', how='left')

# ============= Compute Metrics =============
def compute_summary_metric(row):
    src_txts = ['\n'.join([row[k] for k in parse_spec(args.metric_src_field_specs)])]
    tgt_txts = ['\n'.join([row[k] for k in parse_spec(args.metric_tgt_field_specs)])]
    gen_txts = [row['output']]

    # rouge
    rouge_scorer = RougeMetric()
    rouge_dict = rouge_scorer(gen_txts, tgt_txts)

    for k,v in rouge_dict['score_details_per_sample'].items():# rouge-1, rouge-2, rouge-l
        row[k] = v[0]

    # # bleu-4
    # bleu_scorer = BleuMetric()
    # bleu_dict = bleu_scorer(gen_txts, tgt_txts)

    # for k,v in bleu_dict['score_details_per_sample'].items():# bleu-4
    #     row[k] = v[0]

    # # bert score
    # bert_scorer = ZHBertScoreMetric()
    # bs_dict = bert_scorer(gen_txts, tgt_txts)

    # row['bs_f1'] = bs_dict['score_details_per_sample']['F1'][0]
    # row['bs_p'] = bs_dict['score_details_per_sample']['P'][0]

    # factkb
    factkb_scorer = FactKBMetric()

    row['factkb_ref'] = factkb_scorer(src_txts, tgt_txts)['score_details_per_sample']['factkb'][0]
    row['factkb_gen'] = factkb_scorer(src_txts, gen_txts)['score_details_per_sample']['factkb'][0]

    return row

tqdm.pandas()
result_df = result_df.progress_apply(compute_summary_metric, axis=1)

# ============= Save Results =============
output_dir = os.path.dirname(args.prediction_set_path)
dump_args(args, output_dir, "score_args.json")
output_file_path = os.path.join(output_dir, "metric_scores.jsonl")

metric_only_result_df = result_df.drop(columns=['metric_src', 'metric_ref', 'metric_gen'])
metric_only_result_df.to_json(output_file_path, lines=True, orient='records')
print(f">> Metric outputs saved to {output_file_path}")
