import unittest
import numpy as np
from src import LogisticRegressionModel, NaiveBayesModel


class TestLogisticRegressionModel(unittest.TestCase):
    def setUp(self):
        """Create an object, variables for the unit test functions."""
        self.model = LogisticRegressionModel()
        # Dummy data: 4 samples,
        # including 2 features (X_train, X_test) and 2 targets (y_train, y_test)
        self.X_train = np.array([[1, 2], [2, 1], [1, 0], [0, 1]])
        self.y_train = np.array([0, 1, 0, 1])
        self.X_test = np.array([[1, 1], [0, 0]])
        self.y_test = np.array([0, 1])

    def test_train(self):
        """Unit test for training Logistic Regression model."""
        self.model.train(self.X_train, self.y_train)
        self.assertTrue(hasattr(self.model.classifier, "coef_"))
        self.assertTrue(hasattr(self.model.classifier, "classes_"))

    def test_predict(self):
        """Unit test for making predictions."""
        self.model.train(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        self.assertEqual(y_pred.shape, self.y_test.shape)

    def test_evaluate(self):
        """Unit test for evaluating Logistic Regression model."""
        self.model.train(self.X_train, self.y_train)
        metrics = self.model.evaluate(self.X_test, self.y_test)
        self.assertIsInstance(metrics['accuracy'], float)
        self.assertIsInstance(metrics['precision_macro'], (float, np.ndarray))
        self.assertIsInstance(metrics['recall_macro'], (float, np.ndarray))
        self.assertIsInstance(metrics['f1_macro'], (float, np.ndarray))
        self.assertIsInstance(metrics['f1_weighted'], (float, np.ndarray))
        self.assertIsInstance(metrics['confusion_matrix'], np.ndarray)
        self.assertIsInstance(metrics['classification_report'], (str, dict))
    
    def test_scale_feature(self):
        """Unit test for feature scaling."""
        scaled_feature_train, scaled_feature_test = self.model.scale_feature(
            self.X_train, self.X_test
        )
        self.assertEqual(scaled_feature_train.shape, self.X_train.shape)
        self.assertEqual(scaled_feature_test.shape, self.X_test.shape)


class TestNaiveBayesModel(unittest.TestCase):
    def setUp(self):
        """Create an object, variables for the unit test functions."""
        self.model = NaiveBayesModel()
        # Dummy data: 4 samples,
        # including 2 features (X_train, X_test) and 2 targets (y_train, y_test)
        self.X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        self.y_train = np.array([0, 1, 0, 1])
        self.X_test = np.array([[4, 4], [5, 5]])
        self.y_test = np.array([1, 0])

    def test_train(self):
        """Unit test for training Naive Bayes model."""
        self.model.train(self.X_train, self.y_train)
        self.assertTrue(hasattr(self.model.classifier, "class_count_"))
        self.assertTrue(hasattr(self.model.classifier, "classes_"))

    def test_predict(self):
        """Unit test for making predictions."""
        self.model.train(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        self.assertEqual(y_pred.shape, self.y_test.shape)

    def test_evaluate(self):
        """Unit test for evaluating Naive Bayes model."""
        self.model.train(self.X_train, self.y_train)
        metrics = self.model.evaluate(self.X_test, self.y_test)
        self.assertIsInstance(metrics['accuracy'], float)
        self.assertIsInstance(metrics['precision_macro'], (float, np.ndarray))
        self.assertIsInstance(metrics['recall_macro'], (float, np.ndarray))
        self.assertIsInstance(metrics['f1_macro'], (float, np.ndarray))
        self.assertIsInstance(metrics['f1_weighted'], (float, np.ndarray))
        self.assertIsInstance(metrics['confusion_matrix'], np.ndarray)
        self.assertIsInstance(metrics['classification_report'], (str, dict))


if __name__ == "__main__":
    unittest.main()
