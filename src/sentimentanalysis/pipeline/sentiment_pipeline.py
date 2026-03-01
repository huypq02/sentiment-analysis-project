from sentimentanalysis.data import Preprocessor


class SentimentPipeline:
    def __init__(self, extractor_strategy, model_strategy):
        """
        Initialize classifier with feature extractor and model.
        
        :param extractor_strategy: Feature extraction strategy (TFIDFExtractor, etc.)
        :type extractor_strategy: BaseFeatureExtractor
        :param model_strategy: Model strategy (LogisticRegressionModel, etc.)
        :type model_strategy: SentimentModel
        """
        self.preprocessor = Preprocessor()
        self.extractor = extractor_strategy
        self.model = model_strategy
    
    def set_extractor(self, extractor_strategy):
        """
        Set a extractor.
        
        :param extractor_strategy: Feature extraction strategy
        :type extractor_strategy: BaseFeatureExtractor
        """
        self.extractor = extractor_strategy

    def set_model(self, model_strategy):
        """
        Set a model.
        
        :param model_strategy: Model strategy
        :type model_strategy: SentimentModel
        """
        self.model = model_strategy

    def train(self, X_train, y_train):
        """
        Train a model.
        
        :param X_train: Training features
        :type X_train: array-like
        :param y_train: Training labels
        :type y_train: array-like
        """
        self.model.train(X_train, y_train)
    
    def predict(self, texts):
        """
        Predict sentiment for texts.
        
        :param texts: Input texts to predict
        :type texts: array-like
        :return: Prediction result
        :rtype: int
        """
        predictions = self.model.predict(texts)
        return predictions[0]

    def evaluate(self, y_pred, y_test):
        """
        Evaluate a model.
        
        :param y_pred: Predicted labels
        :type y_pred: array-like
        :param y_test: True labels
        :type y_test: array-like
        :return: Evaluation metrics
        :rtype: dict
        """
        return self.model.evaluate(y_pred, y_test)
