from data.preprocessor import Preprocessor

class SentimentClassifier:
    def __init__(self, model_strategy):
        """Initialize essential parameters for a model."""
        self.model = model_strategy

    def set_model(self, model_strategy):
        """Set a model."""
        self.model = model_strategy

    def train_model(self, X_train, y_train):
        """Train a model."""
        self.model.train(X_train, y_train)

    def classify(self, text):
        """Classify the provided data."""
        preprocessor = Preprocessor()
        text = preprocessor.preprocess(self, text)
        return self.model.predict(text)