from abc import ABC, abstractmethod

class SentimentModel(ABC):
    @abstractmethod
    def train(self):
        """Train the model"""
        pass

    @abstractmethod
    def predict(self):
        """Make predictions"""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluate the model"""
        pass