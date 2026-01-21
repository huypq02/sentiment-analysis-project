import mlflow, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils import load_config, setup_logging
from src.data import DataLoader, Preprocessor
from src.features import TFIDFExtractor
from src.models import LogisticRegressionModel

logger = setup_logging(__name__)


def main(config_path: str = "config/config.yaml", feature_scaling: bool = False):
    """Train and evaluate a Logistic Regression model on the product reviews dataset with MLflow tracking."""

    mlflow.end_run()
    
    mlflow.set_experiment("sentiment-analysis")

    # Load config
    logger.info("Loading configuration...")
    config = load_config(config_path)
    if config is None:
        logger.error("Failed to load configuration. Exiting.")
        raise RuntimeError("Failed to load configuration.")
    filename = os.path.join(config["file"]["raw_dir"], config["file"]["name"])

    # Load the dataset
    logger.info("Loading data ...")
    loader = DataLoader()
    df = loader.load_csv(filename)

    # Preprocessing
    logger.info("Data preprocessing ...")
    preprocessor = Preprocessor()
    df["reviewText_clean"] = df["reviewText"].apply(preprocessor.preprocess) # Tokens
    texts_cleaned = df["reviewText_clean"].apply(lambda x: " ".join(x))      # Texts
    labels = df["rating"]
    logger.info("Training on SENTIMENT labels")
    logger.info(f"Sentiment distribution: {labels.value_counts().to_dict()}")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts_cleaned, labels, test_size=0.2, random_state=42
    )

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 8888,
        "C": 1.0,  # Regularization strength (smaller = stronger regularization)
        "class_weight": "balanced",  # Handle class imbalance automatically
    }

    # Start an MLflow run
    with mlflow.start_run() as run:
        # Log configuration and hyperparameters
        mlflow.log_params(params)

        # Extractor Features with n-grams to capture phrases
        logger.info("Implementing the extractor feature...")
        extractor = TFIDFExtractor(
            max_features=5000,
            ngram_range=(1, 2),  # Unigrams + bigrams (e.g., "not bad")
            min_df=2,
            max_df=0.9
        )
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("ngram_range", "(1,2)")
        
        feature_train = extractor.fit_transform(X_train)
        feature_test = extractor.transform(X_test)
        logger.info(f"Feature matrix shape: {feature_train.shape}")

        # Model strategy
        logger.info("Implementing the model...")
        model = LogisticRegressionModel(params)

        # Log the model
        mlflow.sklearn.log_model(sk_model=model.classifier, 
                                              name="model",
                                              registered_model_name="sentiment-classifier")

        if (
            feature_scaling
        ):  # TODO: Consider if-else with the model no need feature scaling
            feature_train_scaled, feature_test_scaled = model.scale_feature(
                feature_train, feature_test
            )  # Feature scaling
            model.train(feature_train_scaled, y_train)  # Train data on the model
        else:
            model.train(feature_train, y_train)

        # Predict on the test set, compute and log comprehensive metrics
        y_pred = model.predict(feature_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_macro", precision_macro)
        mlflow.log_metric("recall_macro", recall_macro)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 (macro): {f1_macro:.4f}")
        logger.info(f"Precision (macro): {precision_macro:.4f}")
        logger.info(f"Recall (macro): {recall_macro:.4f}")

        # Optional: Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Sentiment Info", "Basic LR model for the product review data")

    run_id = run.info.run_id
    print(
        f"run_id: {run_id}; lifecycle_stage: {mlflow.get_run(run_id).info.lifecycle_stage}"
    )

if __name__ == "__main__":
    main()
