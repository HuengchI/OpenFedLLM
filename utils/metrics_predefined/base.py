from abc import ABC, abstractclassmethod


class MetricCalculator(ABC):
    """
        For subclass authors:
        This class is designed to be used as a callable, please ensure that each call to __call__ is stateless.
    """
    @abstractclassmethod
    def __call__(self, decoded_preds, decoded_labels) -> dict:
        pass
