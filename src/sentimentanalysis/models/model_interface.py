from abc import ABC, abstractmethod


class SentimentModel(ABC):
    @abstractmethod
    def train(self, features, labels):
        """
        Train the model using the provided features and labels.
        
        :param features: Training features
        :type features: array-like
        :param labels: Training labels
        :type labels: array-like
        """
        pass

    @abstractmethod
    def predict(self, feature_test):
        """
        Make predictions on the provided data.
        
        :param feature_test: Test features
        :type feature_test: array-like
        :return: Predictions
        :rtype: array-like
        """
        pass

    @abstractmethod
    def evaluate(self, feature_test, label_test):
        """
        Evaluate the model using the provided test data and test labels.
        
        :param feature_test: Test features
        :type feature_test: array-like
        :param label_test: Test labels
        :type label_test: array-like
        :return: Evaluation metrics dictionary
        :rtype: dict
        """
        pass

    def scale_feature(self, features, feature_test):
        """
        Feature scaling the provided features.
        
        :param features: Training features
        :type features: array-like
        :param feature_test: Test features
        :type feature_test: array-like
        :return: Tuple of scaled features (features, feature_test)
        :rtype: tuple
        """
        pass
