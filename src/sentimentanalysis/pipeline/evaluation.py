import os
import joblib
from .sentiment_pipeline import SentimentPipeline
from sentimentanalysis.utils import load_config, setup_logging
from sentimentanalysis.config import DEFAULT_CONFIG_PATH

logger = setup_logging(__name__)


def evaluate(
        pipeline, 
        feature_test, 
        label_test
):
    """
    Evaluate model performance (accuracy, f1, confusion matrix...).

    :param pipeline: Sentiment pipeline containing the trained model
    :type pipeline: SentimentPipeline
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
        logger.debug("Using config path for evaluation output: %s", config_path)
        logger.debug("Evaluation feature shape: %s", getattr(feature_test, "shape", "unknown"))
        logger.debug("Evaluation label count: %d", len(label_test))
        
        logger.info("Evaluating model...")
        metrics = pipeline.evaluate(feature_test, label_test)

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
        
    except Exception as e:
        logger.exception(f"Unexpected error in evaluation: {e}")
        raise RuntimeError("Evaluation failed")
    
    return metrics


def evaluate_saved_model(
        config_path=None, 
        data_path=None,
        test_size=0.2,
        random_state=42,
        text_column="review", 
        label_column="sentiment"
):
    """
    Load saved model and evaluate on test data.
    
    :param config_path: Path to configuration file
    :type config_path: str
    :param data_path: Path to test data (if None, uses train/test split from config)
    :type data_path: str
    :param text_column: Name of text column
    :type text_column: str
    :param label_column: Name of label column
    :type label_column: str
    :return: Evaluation metrics
    :rtype: dict
    """
    config_path = config_path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)
    config = load_config(config_path)
    logger.debug("Using config path: %s", config_path)
    
    # Load saved model and extractor
    model_path = config["models"]["model"]
    extractor_path = config["models"]["extractor"]
    
    if not os.path.exists(model_path) or not os.path.exists(extractor_path):
        raise FileNotFoundError("Trained model not found. Run training first.")
    
    logger.info("Loading saved model and extractor...")
    model = joblib.load(model_path)
    extractor = joblib.load(extractor_path)
    pipeline = SentimentPipeline(extractor, model)

    # Load data
    data_path = data_path or os.path.join(config["dataset"]["raw_dir"], config["dataset"]["file"])
    df = pipeline.load_dataset(data_path)

    # Preprocess texts and extract labels
    texts_cleaned, labels = pipeline.extract_texts_and_labels(
        df=df,
        text_column=text_column,
        label_column=label_column,
    )

    # Split test set
    _, X_test, _, y_test = pipeline.split_data(
        texts_cleaned=texts_cleaned,
        labels=labels,
        test_size=test_size,
        random_state=random_state,
    )

    # Transform features
    feature_test = pipeline.transform(X_test)

    return evaluate(pipeline, feature_test, y_test)
