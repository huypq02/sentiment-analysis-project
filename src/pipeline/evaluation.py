from .training import train
from src.utils import setup_logging

logger = setup_logging(__name__)


def evaluate(model=None, y_test=None, config=None):
    """Evaluate model performance (accuracy, f1, confusion matrix...)."""

    try:
        if model is None:
            # 1-7 Implement the extractor features and model
            model, _, feature_test_scaled, y_test, config = train()

        # 8. Evaluation
        metrics = model.evaluate(feature_test_scaled, y_test)

        # 9. Log results
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
        return None


if __name__ == "__main__":
    evaluate()
