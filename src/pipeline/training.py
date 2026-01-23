import os
import joblib
from sklearn.model_selection import train_test_split
from src.data import DataLoader, Preprocessor
from src.features import TFIDFExtractor
from src.models import LogisticRegressionModel
from src.utils import load_config, setup_logging

logger = setup_logging(__name__)


def train(config_path="config/config.yaml", feature_scaling: bool = False):
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
        logger.info(f"Sentiment distribution: {labels.value_counts().to_dict()}")

        # 4. Split
        logger.info("Splitting dataset...")
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(
            texts_cleaned, labels, test_size=0.2, random_state=0
        )

        # 5. Extractor Features with n-grams
        logger.info("Implementing the extractor feature...")
        # TODO: consider loading config of a specific feature
        extractor = TFIDFExtractor(
            max_features=5000,
            ngram_range=(1, 2),  # Unigrams + bigrams to capture phrases like "not bad"
            min_df=2,
            max_df=0.9
        )
        feature_train = extractor.fit_transform(X_train)
        feature_test = extractor.transform(X_test)
        logger.info(f"Feature matrix shape: train={feature_train.shape}, test={feature_test.shape}")

        # 6. Model strategy with class balancing
        logger.info("Implementing the model...")
        # TODO: consider loading config of a specific model
        model = LogisticRegressionModel({
            'random_state': 0,
            'max_iter': 1000,
            'class_weight': 'balanced'  # Handle class imbalance
        })

        if (
            feature_scaling
        ):  # TODO: Consider if-else with the model no need feature scaling
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
