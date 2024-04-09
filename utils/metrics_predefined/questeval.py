from questeval.questeval_metric import QuestEval
from utils.metrics_predefined.base import MetricCalculator

preloaded_questeval_scorer_model = None

class QuestEvalMetric(MetricCalculator):
    def __init__(self) -> None:
        super().__init__()

        global preloaded_questeval_scorer_model
        if preloaded_questeval_scorer_model is None:
            print("loading preloaded_questeval_scorer_model...")
            preloaded_questeval_scorer_model = QuestEval()

        self.scorer = preloaded_questeval_scorer_model

    def __call__(self, articles: list[str], summaries: list[str], ref_summaries: list[list[str]]=None):
        """
        Return a score dict containing the averaged metrics for all given (pred, label) pairs
        """
        assert isinstance(articles, list) and isinstance(summaries, list)
        if ref_summaries is not None:
            assert isinstance(ref_summaries, list) and all(isinstance(sublist, list) for sublist in ref_summaries)

        score = self.scorer.corpus_questeval(
            hypothesis=summaries, 
            sources=articles,
            list_references=ref_summaries,
        )

        return {"questeval": score['corpus_score'],
                "score_details_per_sample":{
                    "questeval": score['ex_level_scores'],
                }}