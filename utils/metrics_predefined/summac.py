import numpy as np
from summac.model_summac import SummaCZS, SummaCConv
from utils.metrics_predefined.base import MetricCalculator

preloaded_summac_zs_scorer_model = None
preloaded_summac_conv_scorer_model = None

class SummaCMetric(MetricCalculator):
    def __init__(self) -> None:
        super().__init__()

        global preloaded_summac_zs_scorer_model, preloaded_summac_conv_scorer_model
        if preloaded_summac_zs_scorer_model is None:
            print("loading preloaded_summac_scorer_model...")
            preloaded_summac_zs_scorer_model = SummaCZS(granularity="sentence", model_name="vitc", device="cuda")
            preloaded_summac_conv_scorer_model = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")

    def __call__(self, articles: list, summaries: list):
        """
        Return a score dict containing the averaged metrics for all given (pred, label) pairs
        """
        assert isinstance(articles, list) and isinstance(summaries, list)

        score_zs = preloaded_summac_zs_scorer_model.score(articles, summaries)
        score_conv = preloaded_summac_conv_scorer_model.score(articles, summaries)

        return {"SummaC-ZS": np.mean(score_zs['scores']),
                "SummaC-Conv": np.mean(score_conv['scores']),
                "score_details_per_sample":{
                    "SummaC-ZS": score_zs['scores'],
                    "SummaC-Conv": score_conv['scores'],
                }}