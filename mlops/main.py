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
    """Train and evaluate a Logistic Regression model on the Iris dataset with MLflow tracking."""
    mlflow.set_experiment("Sentiment Analysis")

    # Load config
    logger.info("Loading configuration...")
    mlflow.log_params("config_file", config_path)
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

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts_cleaned, labels, test_size=0.2, random_state=42
    )

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Train the model
        lr = LogisticRegression(**params)
        lr.fit(X_train, y_train)

        # Extractor Features
        logger.info("Implementing the extractor feature...")
        extractor = TFIDFExtractor()
        feature_train = extractor.fit_transform(X_train)
        feature_test = extractor.transform(X_test)

        # Model strategy
        logger.info("Implementing the model...")
        model = LogisticRegressionModel()

        # Log the model
        model_info = mlflow.sklearn.log_model(sk_model=model, name="sentiment_model")

        if (
            feature_scaling
        ):  # TODO: Consider if-else with the model no need feature scaling
            feature_train_scaled, feature_test_scaled = model.scale_feature(
                feature_train, feature_test
            )  # Feature scaling
            model.train(feature_train_scaled, y_train)  # Train data on the model
        else:
            model.train(feature_train, y_train)

        # Predict on the test set, compute and log the loss metric
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Optional: Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic LR model for the product review data")

if __name__ == "__main__":
    main()