from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report
)
from .model_interface import SentimentModel


class LogisticRegressionModel(SentimentModel):
    def __init__(
            self, 
            params: dict = {
                "random_state": 0
            }
    ):
        """Initialize the Logistic Regression and Standard scaler."""
        self.classifier = LogisticRegression(**params)
        self.scaler = StandardScaler(
            with_mean=False
        )  # with_mean=False works with the sparse matrix (mostly zeros)

    def scale_feature(self, features, feature_test):
        """Feature scaling the provided features."""
        features = self.scaler.fit_transform(features)
        feature_test = self.scaler.transform(feature_test)
        return features, feature_test

    def train(self, features, labels):
        """Train Logistic Regression model"""
        self.classifier.fit(features, labels)

    def predict(self, feature_test):
        """Make predictions"""
        return self.classifier.predict(feature_test)

    def evaluate(self, feature_test, label_test):
        """Evaluate Logistic Regression model with comprehensive metrics"""
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
