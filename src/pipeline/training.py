import os
import joblib
from typing import Optional
from sklearn.model_selection import train_test_split
from src.data import DataLoader, Preprocessor
from src.features import ExtractorFactory
from src.models import ModelFactory
from src.utils import load_config, setup_logging
from src.config import (
    DataParameters, 
    ComponentSelection,
    Hyperparameters,
    TrainingConfiguration,
    FilePaths,
    MLFlowTracking
)

logger = setup_logging(__name__)


def train(
        data_params: DataParameters,
        component_sel: ComponentSelection,
        hyperparams: Hyperparameters,
        training_conf: TrainingConfiguration,
        file_paths: FilePaths,
        mlflow_tracking: MLFlowTracking,
):
    """
    The training pipeline on the model.

    Args:
        data_params (DataParameters): Parameters related to data loading and columns.
        component_sel (ComponentSelection): Selection of feature extractor and model.
        hyperparams (Hyperparameters): Hyperparameters for extractor and model.
        training_conf (TrainingConfiguration): Training configuration such as test size, random state, and feature scaling.
        file_paths (FilePaths): File paths for configuration, models, and other artifacts.
        mlflow_tracking (MLFlowTracking): MLflow tracking configuration for experiment logging.

    Returns:
        tuple: (model, extractor, feature_test_scaled, y_test, config)
    """

    # 1. Load config
    logger.info("Loading configuration...")
    config_path: str = os.environ.get("CONFIG_PATH", file_paths.config_path)
    config = load_config(config_path)
    if config is None:
        logger.error("Failed to load configuration. Exiting.")
        raise RuntimeError("Failed to load configuration.")
    data_params.data_path = os.path.join(config["dataset"]["raw_dir"], config["dataset"]["file"])

    try:
        # 2. Load data
        logger.info("Loading data...")
        # Define a DataLoader's object
        loader = DataLoader()
        # Import dataset
        df = loader.load_csv(data_params.data_path)

        # 3. Preprocessing
        logger.info("Data preprocessing...")
        # Preprocess data
        preprocessor = Preprocessor()
        df["reviewText_clean"] = df[data_params.text_column].apply(preprocessor.preprocess)
        # Convert df['reviewText_clean'] from tokens to string X
        texts_cleaned = df["reviewText_clean"].apply(lambda x: " ".join(x))        
        labels = df[data_params.label_column]
        logger.info(f"Sentiment distribution: {labels.value_counts().to_dict()}")

        # 4. Split
        logger.info("Splitting dataset...")
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(
            texts_cleaned, labels, test_size=training_conf.test_size, random_state=training_conf.random_state
        )

        # 5. Extractor Features with n-grams
        logger.info("Implementing the extractor feature...")
        extractor = ExtractorFactory.create_extractor(
            extractor_name=component_sel.extractor_name,
            params=hyperparams.extractor_params
        )
        feature_train = extractor.fit_transform(X_train)
        feature_test = extractor.transform(X_test)
        logger.info(f"Feature matrix shape: train={feature_train.shape}, test={feature_test.shape}")

        # 6. Model strategy with class balancing
        logger.info("Implementing the model...")
        model = ModelFactory.create_model(
            model_name=component_sel.model_name,
            params=hyperparams.model_params
        )
        if (
            training_conf.feature_scaling
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
        return RuntimeError("Training pipeline failed")

    # 7. Save model and feature extractor
    # Create new folder with the name 'models' if it doesn't exist
    os.makedirs(config["models"]["dir"], exist_ok=True)

    logger.info("Saving model and extractor...")
    # Dump files
    joblib.dump(model, config["models"]["model"])
    joblib.dump(extractor, config["models"]["extractor"])

    return model, extractor, feature_test_scaled, y_test, config


if __name__ == "__main__":
    train(
        data_params=DataParameters(),
        component_sel=ComponentSelection(),
        hyperparams=Hyperparameters(
            extractor_params={
                "max_features": 5000,
                "ngram_range": (1, 2),  # Unigrams + bigrams to capture phrases like "not bad"
                "min_df": 1,
                "max_df": 0.9
            },
            model_params={
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": 8888,
                "C": 1.0,  # Regularization strength (smaller = stronger regularization)
                "class_weight": "balanced",  # Handle class imbalance automatically
            }
        ),
        training_conf=TrainingConfiguration(),
        file_paths=FilePaths(),
        mlflow_tracking=MLFlowTracking()
    )
