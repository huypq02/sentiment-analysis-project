from typing import Optional
from . import LogisticRegressionModel, NaiveBayesModel

class ModelFactory():
    @staticmethod
    def create_model(model_name: str, params: Optional[dict]):
        """
        Create a model.

        :param model_name: The name of the model to create (Supported: 'logreg', 'naive_bayes')
        :type model_name: str
        :param params: Parameters to initialize the model (If None, default parameters are used)
        :type params: Optional[dict]
        :return: An instance of the specified model
        :rtype: SentimentModel
        :raises ValueError: If an unknown model_name is provided
        """
        if model_name == "logreg":
            if not params:
                return LogisticRegressionModel()
            return LogisticRegressionModel(params)
        elif model_name == "naive_bayes":
            if not params:
                return NaiveBayesModel()
            return NaiveBayesModel(params)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
