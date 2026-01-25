import os
import joblib
from sklearn.model_selection import train_test_split
from src.data import DataLoader, Preprocessor
from src.features import ExtractorFactory
from src.models import ModelFactory
from src.utils import load_config, setup_logging

logger = setup_logging(__name__)


def train(
        # Data parameters
        data_path: str = None,
        text_column: str = "reviewText",
        label_column: str = "rating",

        # Component selection
        extractor_name: str = "tfidf",
        model_name: str = "logreg",

        # Hyperparameters
        extractor_params: dict = None,
        model_params: dict = None,

        # Training configuration
        test_size: float = 0.2,
        random_state: int = 0,
        feature_scaling: bool = False,

        # File paths
        config_path: str = "config/config.yaml",
        save_dir: str = None,

        # MLFlow tracking
        mlflow_tracking: bool = True,
        experiment_name: str = "sentiment-analysis",
        run_name: str = None,
        tags=None,
):
    """The training pipeline on the model"""
    # 1. Load config
    logger.info("Loading configuration...")
    config = load_config(config_path)
    if config is None:
        logger.error("Failed to load configuration. Exiting.")
        raise RuntimeError("Failed to load configuration.")
    data_path = os.path.join(config["dataset"]["raw_dir"], config["dataset"]["file"])

    try:
        # 2. Load data
        logger.info("Loading data...")
        # Define a DataLoader's object
        loader = DataLoader()
        # Import dataset
        df = loader.load_csv(data_path)

        # 3. Preprocessing
        logger.info("Data preprocessing...")
        # Preprocess data
        preprocessor = Preprocessor()
        df["reviewText_clean"] = df[text_column].apply(preprocessor.preprocess)
        # Convert df['reviewText_clean'] from tokens to string X
        texts_cleaned = df["reviewText_clean"].apply(lambda x: " ".join(x))        
        labels = df[label_column]
        logger.info(f"Sentiment distribution: {labels.value_counts().to_dict()}")

        # 4. Split
        logger.info("Splitting dataset...")
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(
            texts_cleaned, labels, test_size=test_size, random_state=random_state
        )

        # 5. Extractor Features with n-grams
        logger.info("Implementing the extractor feature...")
        extractor = ExtractorFactory.create_extractor(extractor_name)
        feature_train = extractor.fit_transform(X_train)
        feature_test = extractor.transform(X_test)
        logger.info(f"Feature matrix shape: train={feature_train.shape}, test={feature_test.shape}")

        # 6. Model strategy with class balancing
        logger.info("Implementing the model...")
        model = ModelFactory.create_model(model_name)
        if (
            feature_scaling
        ):
            feature_train_scaled, feature_test_scaled = model.scale_feature(
                feature_train, feature_test
            )  # Feature scaling
            model.train(feature_train_scaled, y_train)  # Train data on the model
        else:
            model.train(feature_train, y_train)
            feature_test_scaled = feature_test  # Use unscaled features

    except Exception as e:
        logger.exception(f"Unexpected error in training pipeline: {e}")
        return None

    # 7. Save model and feature extractor
    # Create new folder with the name 'models' if it doesn't exist
    os.makedirs(config["models"]["dir"], exist_ok=True)

    logger.info("Saving model and extractor...")
    # Dump files
    joblib.dump(model, config["models"]["model"])
    joblib.dump(extractor, config["models"]["extractor"])

    return model, extractor, feature_test_scaled, y_test, config


if __name__ == "__main__":
    train()
