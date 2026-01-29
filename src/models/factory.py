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
            if params is None:
                params = {
                    'random_state': 0,
                    'max_iter': 1000,
                    'class_weight': 'balanced'  # Handle class imbalance
                }
            return LogisticRegressionModel(params)
        elif model_name == "naive_bayes":
            if params is None:
                params = {
                    'alpha': 1.0,           # Smoothing (0=no smoothing, 1=Laplace, higher=more smoothing)
                    'fit_prior': True,
                    'class_prior': None
                }
            return NaiveBayesModel(params)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
