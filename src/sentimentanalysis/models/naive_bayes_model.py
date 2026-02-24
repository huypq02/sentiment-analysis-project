from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
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
        self.classifier = MultinomialNB(**params)
        self.scaler = StandardScaler(
            with_mean=False
        )  # with_mean=False works with the sparse matrix (mostly zeros)

    def train(self, features, labels):
        """Train the provided data on the Naive Bayes model."""
        self.classifier.fit(features, labels)

    def predict(self, feature_test):
        """Make a prediction."""
        return self.classifier.predict(feature_test)

    def evaluate(self, feature_test, label_test):
        """Evaluate Naive Bayes model"""
        y_pred = self.classifier.predict(feature_test)
        
        return {
            'confusion_matrix': confusion_matrix(label_test, y_pred),
            'accuracy': accuracy_score(label_test, y_pred),
            'precision_macro': precision_score(label_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(label_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(label_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(label_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(label_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(label_test, y_pred, average='weighted', zero_division=0),
            'classification_report': classification_report(label_test, y_pred, zero_division=0)
        }
