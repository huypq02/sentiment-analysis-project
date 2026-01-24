from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report
)
from .model_interface import SentimentModel


class NaiveBayesModel(SentimentModel):
    def __init__(
            self,
            params: dict = {}
    ):
        """Initialize the Naive Bayes model and Standard scaler."""
        self.classifier = GaussianNB(**params)
        self.scaler = StandardScaler(
            with_mean=False
        )  # with_mean=False works with the sparse matrix (mostly zeros)

    def scale_feature(self, data, test_data):
        """Feature scaling the provided features.
        This method does NOT modify the input arrays in-place.
        Returns scaled copies of the input data and test_data."""
        data = self.scaler.fit_transform(data)
        test_data = self.scaler.transform(test_data)
        return data, test_data

    def train(self, data, labels):
        """Train the provided data on the Naive Bayes model."""
        self.classifier.fit(data, labels)

    def predict(self, test_data):
        """Make a prediction."""
        return self.classifier.predict(test_data)

    def evaluate(self, test_data, test_labels):
        """Evaluate Naive Bayes model"""
        y_pred = self.classifier.predict(test_data)
        
        return {
            'confusion_matrix': confusion_matrix(test_labels, y_pred),
            'accuracy': accuracy_score(test_labels, y_pred),
            'precision_macro': precision_score(test_labels, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(test_labels, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(test_labels, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(test_labels, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(test_labels, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(test_labels, y_pred, average='weighted', zero_division=0),
            'classification_report': classification_report(test_labels, y_pred, zero_division=0)
        }
