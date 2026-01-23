from src.data import Preprocessor


class SentimentPipeline:
    def __init__(self, extractor_strategy, model_strategy):
        """
        Initialize classifier with feature extractor and model.
        
        Args:
            extractor_strategy: Feature extraction strategy (TFIDFExtractor, etc.)
            model_strategy: Model strategy (LogisticRegressionModel, etc.)
        """
        self.preprocessor = Preprocessor()
        self.extractor = extractor_strategy
        self.model = model_strategy
    
    def set_extractor(self, extractor_strategy):
        """Set a extractor."""
        self.extractor = extractor_strategy

    def set_model(self, model_strategy):
        """Set a model."""
        self.model = model_strategy

    def train(self, X_train, y_train):
        """Train a model."""
        self.model.train(X_train, y_train)
    
    def predict(self, texts):
        """Train a model."""
        predictions = self.model.predict(texts)
        return predictions[0]

    def evaluate(self, y_pred, y_test):
        """Evaluate a model."""
        return self.model.evaluate(y_pred, y_test)
