from bert_score import BERTScorer
from utils.metrics_predefined.base import MetricCalculator

preloaded_bert_scorer_model = None

class ZHBertScoreMetric(MetricCalculator):
    def __init__(self) -> None:
        super().__init__()

        global preloaded_bert_scorer_model
        if preloaded_bert_scorer_model is None:
            print("loading preloaded_bert_scorer_model...")
            preloaded_bert_scorer_model = BERTScorer(lang='zh', rescale_with_baseline=True, device='cuda')

        self.scorer = preloaded_bert_scorer_model

    def __call__(self, decoded_preds, decoded_labels):
        """
        Return a score dict containing the averaged metrics for all given (pred, label) pairs
        """
        P,R,F1 = self.scorer.score(decoded_preds, decoded_labels)
        return {"P": P.mean().item(),
                "R": R.mean().item(),
                "F1": F1.mean().item(),
                "score_details_per_sample":{
                    "P": [v.item() for v in P],
                    "R": [v.item() for v in R],
                    "F1": [v.item() for v in F1],
                }}