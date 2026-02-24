from typing import Optional
from . import LogisticRegressionModel, NaiveBayesModel

class ModelFactory():
    @staticmethod
    def create_model(model_name: str, params: Optional[dict]):
        """
        Create a model.

        Args:
            model_name (str): The name of the model to create. Supported values are 'logreg' and 'naive_bayes'.
            params (Optional[dict]): Parameters to initialize the model. If None, default parameters are used.

        Returns:
            An instance of the specified model.

        Raises:
            ValueError: If an unknown model_name is provided.
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
