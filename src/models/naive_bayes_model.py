from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from .model_interface import SentimentModel

class NaiveBayesModel(SentimentModel):
    def __init__(self):
        """Initialize the Naive Bayes model and Standard scaler."""
        self.classifier = GaussianNB()
        self.scaler = StandardScaler()

    def train(self, data, labels):
        """Train the provided data on the Naive Bayes model."""
        self.classifier.fit(data, labels)
    
    def scale_feature(self, data, test_data):
        """Feature scaling the provided features.
        This method does NOT modify the input arrays in-place.
        Returns scaled copies of the input data and test_data."""     
        data = self.scaler.fit_transform(data)
        test_data = self.scaler.transform(test_data)
        return data, test_data
    
    def predict(self, test_data):
        """Make a prediction."""
        return self.classifier.predict(test_data)
    
    def evaluate(self, test_data, test_labels):
        """Evaluate Naive Bayes model"""
        y_pred = self.classifier.predict(test_data)
        cm = confusion_matrix(test_labels, y_pred)  # Confusion matrix
        score = accuracy_score(test_labels, y_pred) # Accuracy score
        return cm, score