from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from utils.metrics_predefined.base import MetricCalculator


class BleuMetric(MetricCalculator):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, decoded_preds, decoded_labels) -> dict:
        bleu_scores = []
        for pred, label in zip(decoded_preds, decoded_labels):
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            bleu_scores.append(bleu_score)

        return {"bleu-4":round(np.mean(bleu_scores) * 100, 4),
                "score_details_per_sample":{
                    "bleu-4": [round(v * 100, 4) for v in bleu_scores]
                }}