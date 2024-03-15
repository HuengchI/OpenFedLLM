import jieba
from rouge_chinese import Rouge
import numpy as np
from utils.metrics_predefined.base import MetricCalculator


class RougeMetric(MetricCalculator):
    def __init__(self, logger=None) -> None:
        super().__init__()
        self.scorer = Rouge()
        self.logger = logger
    
    def __call__(self, decoded_preds, decoded_labels):
        assert len(decoded_labels) == len(decoded_preds)

        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = ' '.join(list(jieba.cut(pred)))
            reference = ' '.join(list(jieba.cut(label)))
            try:
                result = self.scorer.get_scores(hypothesis, reference)[0]
            except ValueError as e:
                if self.logger is not None:
                    self.logger.critical(e)
                    self.logger.critical(f"pred: {pred}")
                    self.logger.critical(f"label: {label}")
                rouge_1_scores.append(0.0)
                rouge_2_scores.append(0.0)
                rouge_l_scores.append(0.0)
                continue
            rouge_1_scores.append(round(result['rouge-1']['f']*100,4))
            rouge_2_scores.append(round(result['rouge-2']['f']*100,4))
            rouge_l_scores.append(round(result['rouge-l']['f']*100,4))
            
        return {
            "rouge-1%": np.mean(rouge_1_scores),
            "rouge-2%": np.mean(rouge_2_scores),
            "rouge-l%": np.mean(rouge_l_scores),
            "score_details_per_sample":{
                "rouge-1%": rouge_1_scores,
                "rouge-2%": rouge_2_scores,
                "rouge-l%": rouge_l_scores,
            }
        }