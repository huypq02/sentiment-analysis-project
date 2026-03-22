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

    def preprocess_texts(self, texts):
        """
        Preprocess raw texts into cleaned strings.

        :param texts: Input text or texts
        :type texts: str | array-like
        :return: Cleaned texts ready for feature extraction
        :rtype: list[str]
        """
        if isinstance(texts, str):
            texts = [texts]

        cleaned_texts = []
        for text in texts:
            tokens = self.preprocessor.preprocess(text)
            cleaned_texts.append(" ".join(tokens))

        return cleaned_texts

    def transform(self, texts):
        """
        Transform raw texts into model-ready features.

        :param texts: Input text or texts
        :type texts: str | array-like
        :return: Transformed features
        :rtype: array-like
        """
        cleaned_texts = self.preprocess_texts(texts)
        features = self.extractor.transform(cleaned_texts)

        if getattr(self.model, "scaler", None) is not None:
            features = self.model.scaler.transform(features)

        return features
    
    def predict(self, texts):
        """
        Predict sentiment for texts.
        
        :param texts: Input texts to predict
        :type texts: array-like
        :return: Prediction result
        :rtype: int
        """
        features = self.transform(texts)
        predictions = self.model.predict(features)

        if isinstance(texts, str):
            return predictions[0]

        return predictions

    def evaluate(self, feature_test, label_test):
        """
        Evaluate a model.
        
        :param feature_test: Features
        :type feature_test: array-like
        :param label_test: True labels
        :type label_test: array-like
        :return: Evaluation metrics
        :rtype: dict
        """
        return self.model.evaluate(feature_test, label_test)
