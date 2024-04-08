import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.metrics_predefined.base import MetricCalculator

preloaded_factkb_model = None
preloaded_factkb_tokenizer = None

class FactKBMetric(MetricCalculator):
    def __init__(self) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        global preloaded_factkb_model, preloaded_factkb_tokenizer
        if preloaded_factkb_model is None:
            print("loading preloaded_factkb_model...")
            preloaded_factkb_model = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", num_labels = 2).to(self.device)
            preloaded_factkb_tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True)


    def __call__(self, articles: list[str], summaries: list[str]):
        """
        Return a score dict containing the averaged metrics for all given (pred, label) pairs
        """
        assert isinstance(articles, list) and isinstance(summaries, list)

        input = list(zip(summaries, articles))

        tokens = preloaded_factkb_tokenizer(input, return_tensors="pt", padding="longest", truncation=True)
        result = torch.softmax(
            preloaded_factkb_model(**{k:torch.as_tensor(v).to(self.device) for k,v in tokens.items()}).logits,
                dim = 1)
        factkb_scores = result[:,1]

        return {"factkb": factkb_scores.mean().item(),
                "score_details_per_sample":{
                    "factkb": factkb_scores.tolist(),
                }}