from abc import ABC, abstractmethod


class SentimentModel(ABC):
    @abstractmethod
    def train(self, features, labels):
        """Train the model using the provided features and labels."""
        pass

    @abstractmethod
    def predict(self, feature_test):
        """Make predictions on the provided data."""
        pass

    @abstractmethod
    def evaluate(self, feature_test, label_test):
        """Evaluate the model using the provided test data and test labels."""
        pass

    def scale_feature(self, features, feature_test):
        """Feature scaling the provided features."""
        pass
