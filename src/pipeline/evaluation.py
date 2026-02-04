import os
from .training import train
from src.utils import load_config, setup_logging
from src.config import DEFAULT_CONFIG_PATH

logger = setup_logging(__name__)


def evaluate(
        evaluate_model, 
        feature_test, 
        label_test
):
    """
    Evaluate model performance (accuracy, f1, confusion matrix...).

    Args:
        model: Trained model object with an evaluate method.
        feature_test: Test feature matrix (vectorized/scaled).
        label_test: True labels for the test set.
        config_path (str, optional): Path to configuration YAML file. Defaults to "config/config.yaml".

    Returns:
        dict: Dictionary containing evaluation metrics (accuracy, precision, recall, f1, confusion matrix, classification report).
    """

    try:
        config_path: str = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)
        config = load_config(config_path)
        # 8. Evaluation
        logger.info("Evaluating...")
        metrics = evaluate_model.evaluate(feature_test, label_test)

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
        return RuntimeError("Evaluation failed")


if __name__ == "__main__":
    from src.config import (
    DataParameters, 
    ComponentSelection,
    Hyperparameters,
    TrainingConfiguration,
    FilePaths,
    MLFlowTracking
    )

    model, _, feature_test, label_test = train(
        data_params=DataParameters(),
        component_sel=ComponentSelection(
            extractor_name="tfidf",
            model_name="logreg"
        ),
        hyperparams=Hyperparameters(
            param_grid = {
                "extractor__max_features": [5000, 10000],
                "extractor__ngram_range": [(1,1), (1,2)],
                "extractor__min_df": [2, 5],
                "extractor__max_df": [0.8],
                "extractor__binary": [False],

                "model__solver": ["lbfgs"],
                "model__penalty": ["l2"],
                "model__C": [0.1, 1, 5],
                "model__class_weight": [None, "balanced"],
                "model__max_iter": [1000]
            }
        ),
        training_conf=TrainingConfiguration(),
        file_paths=FilePaths(),
        mlflow_tracking=MLFlowTracking()
    )
    
    evaluate(
        evaluate_model=model,
        feature_test=feature_test,
        label_test=label_test
    )
