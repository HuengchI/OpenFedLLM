"""
e.g.
python score.py --prediction_set_path 'prediction_output/20240315170124.jsonl' \
--test_set_path '../../datasets/DISC-Law-SFT/jud_doc_sum/jud_doc_sum_test_split.jsonl' \
--output_dir 'score_output'
"""

import argparse
import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd

import sys
UTILS_LIB_PATH = "../../"
sys.path.append(os.path.abspath(UTILS_LIB_PATH))
from utils.metrics_predefined import RougeMetric, BleuMetric, ZHBertScoreMetric


parser = argparse.ArgumentParser()
parser.add_argument("--prediction_set_path", type=str, default=None, required=True)
parser.add_argument("--test_set_path", type=str, default=None, required=True)
parser.add_argument("--output_dir", type=str, default=None, required=True)

args = parser.parse_args()


# ============= Load Datasets =============
# prediction_df = pd.read_json('prediction_output/20240315155546.jsonl', lines=True)
prediction_df = pd.read_json(args.prediction_set_path, lines=True)

# test_df = pd.read_json('../../datasets/DISC-Law-SFT/jud_doc_sum/jud_doc_sum_test_split.jsonl', lines=True)
test_df = pd.read_json(args.test_set_path, lines=True)
test_df = test_df.rename(columns={'output': 'ref'})

# ============= Process Datasets =============
prediction_df = prediction_df.rename(columns={'output': 'gen'})
result_df = pd.merge(prediction_df, test_df, on='id', how='left')

# ============= Compute Metrics =============
def compute_summary_metric(row):
    decoded_labels = [row['ref']]
    decoded_preds = [row['gen']]

    # rouge
    rouge_scorer = RougeMetric()
    rouge_dict = rouge_scorer(decoded_preds, decoded_labels)

    for k,v in rouge_dict['score_details_per_sample'].items():# rouge-1, rouge-2, rouge-l
        row[k] = v[0]

    # bleu-4
    bleu_scorer = BleuMetric()
    bleu_dict = bleu_scorer(decoded_preds, decoded_labels)

    for k,v in bleu_dict['score_details_per_sample'].items():# bleu-4
        row[k] = v[0]

    # bert score
    bert_scorer = ZHBertScoreMetric()
    bs_dict = bert_scorer(decoded_preds, decoded_labels)

    row['bs_f1'] = bs_dict['score_details_per_sample']['F1'][0]
    row['bs_p'] = bs_dict['score_details_per_sample']['P'][0]
    
    return row

tqdm.pandas()
result_df = result_df.progress_apply(compute_summary_metric, axis=1)

# ============= Save Results =============
os.makedirs(args.output_dir, exist_ok=True)
output_file_path = os.path.join(args.output_dir, f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jsonl")

metric_only_result_df = result_df.drop(columns=['input', 'ref', 'gen'])
metric_only_result_df.to_json(output_file_path, lines=True, orient='records')
print(f">> Metric outputs saved to {output_file_path}")
