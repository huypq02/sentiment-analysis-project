from abc import ABC, abstractmethod

class SentimentModel(ABC):
    @abstractmethod
    def train(self, data, labels):
        """Train the model using the provided data and labels."""
        pass        

    @abstractmethod
    def scale_feature(self, data, test_data):
        """Feature scaling the provided features."""
        pass

    @abstractmethod
    def predict(self, test_data):
        """Make predictions on the provided data."""
        pass

    @abstractmethod
    def evaluate(self, test_data, test_labels):
        """Evaluate the model using the provided test data and test labels."""
        pass