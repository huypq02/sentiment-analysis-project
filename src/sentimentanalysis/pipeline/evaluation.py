import os
from sentimentanalysis.utils import load_config, setup_logging
from sentimentanalysis.config import DEFAULT_CONFIG_PATH

logger = setup_logging(__name__)


def evaluate(
        evaluate_model, 
        feature_test, 
        label_test
):
    """
    Evaluate model performance (accuracy, f1, confusion matrix...).

    :param evaluate_model: Trained model object with an evaluate method
    :type evaluate_model: SentimentModel
    :param feature_test: Test feature matrix (vectorized/scaled)
    :type feature_test: array-like
    :param label_test: True labels for the test set
    :type label_test: array-like
    :return: Dictionary containing evaluation metrics (accuracy, precision, recall, f1, confusion matrix, classification report)
    :rtype: dict
    """

    try:
        config_path: str = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)
        config = load_config(config_path)
        
        logger.info("Evaluating model...")
        metrics = evaluate_model.evaluate(feature_test, label_test)

        # Save results
        os.makedirs(config["models"]["dir"], exist_ok=True)
        with open(config["models"]["metrics"], "w") as f:
            f.write("=== Model Evaluation Results ===\n\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision (macro): {metrics['precision_macro']:.4f}\n")
            f.write(f"Recall (macro): {metrics['recall_macro']:.4f}\n")
            f.write(f"F1 Score (macro): {metrics['f1_macro']:.4f}\n")
            f.write(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(metrics['confusion_matrix']))
            f.write("\n\n")
            f.write("Classification Report:\n")
            f.write(metrics['classification_report'])
        
        logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Results saved to {config['models']['metrics']}")
        
        return metrics

    except Exception as e:
        logger.exception(f"Unexpected error in evaluation: {e}")
        raise RuntimeError("Evaluation failed")
