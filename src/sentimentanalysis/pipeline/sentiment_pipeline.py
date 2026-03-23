from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sentimentanalysis.data import DataLoader, Preprocessor
from sentimentanalysis.utils import setup_logging


logger = setup_logging(__name__)

class SentimentPipeline:
    def __init__(self, extractor_strategy=None, model_strategy=None):
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
        logger.info("Transforming test data...")
        features = self.extractor.transform(texts)

        scaler = getattr(self.model, "scaler", None)
        if scaler is not None:
            try:
                check_is_fitted(scaler)
                logger.info("Applying feature scaling...")
                features = self.model.scaler.transform(features)
            except NotFittedError:
                logger.info("Skipping feature scaling: scaler exists but is not fitted.")

        return features

    def load_dataset(self, data_path):
        """
        Load a CSV file into a dataset.

        :param data_path: Path to CSV data
        :type data_path: str
        :return: Loaded dataset
        :rtype: pandas.DataFrame
        """
        logger.info("Loading data from %s...", data_path)
        loader = DataLoader()
        return loader.load_csv(data_path)

    def extract_texts_and_labels(self, df, text_column, label_column):
        """
        Extract cleaned texts and labels from a dataframe.

        :param df: Input dataframe
        :type df: pandas.DataFrame
        :param text_column: Name of the text column
        :type text_column: str
        :param label_column: Name of the label column
        :type label_column: str
        :return: Tuple of cleaned texts and labels
        :rtype: tuple
        """
        logger.info("Data preprocessing...")
        texts_cleaned = self.preprocess_texts(df[text_column])
        labels = df[label_column]
        return texts_cleaned, labels

    def split_data(self, texts_cleaned, labels, test_size=0.2, random_state=42):
        """
        Split cleaned texts and labels to get test data.

        :param texts_cleaned: Cleaned texts
        :type texts_cleaned: array-like
        :param labels: Labels
        :type labels: array-like
        :param test_size: Fraction for test split
        :type test_size: float
        :param random_state: Random seed for split reproducibility
        :type random_state: int
        :return: Tuple of test texts and test labels
        :rtype: tuple
        """
        logger.info("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            texts_cleaned,
            labels,
            test_size=test_size,
            random_state=random_state,
        )
        return X_train, X_test, y_train, y_test
    
    def predict(self, texts):
        """
        Predict sentiment for texts.
        
        :param texts: Input texts to predict
        :type texts: str | array-like
        :return: Prediction result
        :rtype: int | array-like
        """
        cleaned = self.preprocess_texts(texts)
        features = self.transform(cleaned)
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
