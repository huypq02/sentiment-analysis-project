from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from .model_interface import SentimentModel

class LogisticRegressionModel(SentimentModel):
    def __init__(self):
        """Initialize the Logistic Regression and Standard scaler."""
        self.classifier = LogisticRegression(random_state=0)
        self.scaler = StandardScaler()

    def train(self, data, labels):
        """Train Logistic Regression model"""
        self.classifier.fit(data, labels)
        
    def scale_feature(self, data, test_data):
        """Feature scaling the provided features.
        This method does NOT modify the input arrays in-place.
        Returns scaled copies of the input data and test_data."""
        data = self.scaler.fit_transform(data)
        test_data = self.scaler.transform(test_data)
        return data, test_data

    def predict(self, test_data):
        """Make predictions"""
        return self.classifier.predict(test_data)

    def evaluate(self, test_data, test_labels):
        """Evaluate Logistic Regression model"""
        y_pred = self.classifier.predict(test_data)
        cm = confusion_matrix(test_labels, y_pred)  # Confusion matrix
        score = accuracy_score(test_labels, y_pred) # Accuracy score
        return cm, score