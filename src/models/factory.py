from . import LogisticRegressionModel, NaiveBayesModel

class ModelFactory():
    @staticmethod
    def create_model(model_name: str):
        """Create a model."""
        if model_name == "logreg":
            return LogisticRegressionModel({
                'random_state': 0,
                'max_iter': 1000,
                'class_weight': 'balanced'  # Handle class imbalance
            })
        elif model_name == "naive_bayes":
            return NaiveBayesModel({
                'alpha': 1.0,           # Smoothing (0=no smoothing, 1=Laplace, higher=more smoothing)
                'fit_prior': True,
                'class_prior': None
            })
        else:
            raise ValueError(f"Unknown model name: {model_name}")
