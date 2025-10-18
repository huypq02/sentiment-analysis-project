from .train_pipeline import train_main
from src.utils.logger import setup_logging

logger = setup_logging(__name__)

def evaluation_main(model=None, y_test=None, config=None):
    """Evaluate model performance (accuracy, f1, confusion matrix...)."""

    if model is None:
        # 1-7 Implement the extractor features and model
        model, _, feature_test_scaled, y_test, config = train_main()

    try:
        # 8. Evaluation
        cm, accuracy = model.evaluate(feature_test_scaled, y_test)
        print("Confusion Matrix:\n", cm)
        print("Accuracy Score:", accuracy)

        # 9. (Optional) Log results
        with open(config['path']['metrics'], "w") as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\nAccuracy Score: ")
            f.write(str(accuracy))
    except Exception as e:
        logger.exception(f'Unexpected error in training pipeline: {e}')
        return None

if __name__ == "__main__":
    evaluation_main()