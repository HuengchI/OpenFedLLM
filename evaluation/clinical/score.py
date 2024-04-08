import os
from typing import Optional
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass, field
from transformers import HfArgumentParser

import sys
THRID_PARTY_LIB_PATH = (
    ".",
    "../../SPEER/", # esg_tools
)
for path in THRID_PARTY_LIB_PATH:
    path = os.path.abspath(path)
    if path not in sys.path:
        sys.path.append(path)

from utils.metrics_predefined import RougeMetric, BleuMetric, ZHBertScoreMetric
from utils import dump_args
from esg_tools import metric_compute_HR, metric_compute_SGR


@dataclass
class ScriptArguments:
    prediction_set_path: Optional[str] = field()
    test_set_path: Optional[str] = field()
    local_rank: Optional[int] = field(default=-1)
    metric_src_field: Optional[str] = field(default='source')
    metric_tgt_field: Optional[str] = field(default='target')

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]


# ============= Load Datasets =============
prediction_df = pd.read_json(args.prediction_set_path, lines=True, dtype=False)
prediction_df = prediction_df.rename(columns={'output': 'metric_gen'})

test_df = pd.read_json(args.test_set_path, lines=True, dtype=False)
test_df = test_df.rename(columns={args.metric_src_field: 'metric_src'}) # for metric calculation
test_df = test_df.rename(columns={args.metric_tgt_field: 'metric_ref'})
test_df = test_df.rename(columns={'example_id': 'id'})
test_df = test_df[['metric_src', 'metric_ref', 'id']]

# ============= Process Datasets =============
result_df = pd.merge(prediction_df, test_df, on='id', how='left')

# ============= Compute Metrics =============
def compute_summary_metric(row):
    src_txts = [row['metric_src']]
    tgt_txts = [row['metric_ref']]
    gen_txts = [row['metric_gen']]

    # rouge
    rouge_scorer = RougeMetric()
    rouge_dict = rouge_scorer(gen_txts, tgt_txts)

    for k,v in rouge_dict['score_details_per_sample'].items():# rouge-1, rouge-2, rouge-l
        row[k] = v[0]

    # bleu-4
    bleu_scorer = BleuMetric()
    bleu_dict = bleu_scorer(gen_txts, tgt_txts)

    for k,v in bleu_dict['score_details_per_sample'].items():# bleu-4
        row[k] = v[0]

    # bert score
    bert_scorer = ZHBertScoreMetric()
    bs_dict = bert_scorer(gen_txts, tgt_txts)

    row['bs_f1'] = bs_dict['score_details_per_sample']['F1'][0]
    row['bs_p'] = bs_dict['score_details_per_sample']['P'][0]

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
