import os
import joblib
from sklearn.model_selection import train_test_split
from src.data import DataLoader, Preprocessor
from src.features import TFIDFExtractor
from src.models import LogisticRegressionModel
from src.utils import load_config, setup_logging

logger = setup_logging(__name__)


def train_main(config_path="config/config.yaml", feature_scaling: bool = False):
    """The training pipeline on the model"""
    # 1. Load config
    logger.info("Loading configuration...")
    config = load_config(config_path)
    if config is None:
        logger.error("Failed to load configuration. Exiting.")
        raise RuntimeError("Failed to load configuration.")
    filename = os.path.join(config["file"]["raw_dir"], config["file"]["name"])

    try:
        # 2. Load data
        logger.info("Loading data...")
        # Define a DataLoader's object
        loader = DataLoader()
        # Import dataset
        df = loader.load_csv(filename)

        # 3. Preprocessing
        logger.info("Data preprocessing...")
        # Preprocess data
        preprocessor = Preprocessor()
        df["reviewText_clean"] = df["reviewText"].apply(preprocessor.preprocess)
        # Convert df['reviewText_clean'] from tokens to string X
        texts_cleaned = df["reviewText_clean"].apply(lambda x: " ".join(x))
        labels = df["rating"]

        # 4. Split
        logger.info("Splitting dataset...")
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(
            texts_cleaned, labels, test_size=0.2, random_state=0
        )

        # 5. Extractor Features
        logger.info("Implementing the extractor feature...")
        # TODO: consider loading config of a specific feature
        extractor = TFIDFExtractor()
        feature_train = extractor.fit_transform(X_train)
        feature_test = extractor.transform(X_test)

        # 6. Model strategy
        logger.info("Implementing the model...")
        # TODO: consider loading config of a specific model
        model = LogisticRegressionModel()

        if (
            feature_scaling
        ):  # TODO: Consider if-else with the model no need feature scaling
            feature_train_scaled, feature_test_scaled = model.scale_feature(
                feature_train, feature_test
            )  # Feature scaling
            model.train(feature_train_scaled, y_train)  # Train data on the model
        else:
            model.train(feature_train, y_train)

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

    return model, extractor, None, y_test, config


if __name__ == "__main__":
    train_main()
