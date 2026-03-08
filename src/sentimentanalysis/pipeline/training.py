import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sentimentanalysis.data import DataLoader, Preprocessor
from sentimentanalysis.features import ExtractorFactory
from sentimentanalysis.models import ModelFactory
from sentimentanalysis.utils import load_config, setup_logging
from sentimentanalysis.config import (
    DataParameters, 
    ComponentSelection,
    Hyperparameters,
    TrainingConfiguration,
    FilePaths,
    MLFlowTracking
)
from .evaluation import evaluate

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

    :param data_params: Parameters related to data loading and columns
    :type data_params: DataParameters
    :param component_sel: Selection of feature extractor and model
    :type component_sel: ComponentSelection
    :param hyperparams: Hyperparameters for extractor and model
    :type hyperparams: Hyperparameters
    :param training_conf: Training configuration such as test size, random state, and feature scaling
    :type training_conf: TrainingConfiguration
    :param file_paths: File paths for configuration, models, and other artifacts
    :type file_paths: FilePaths
    :param mlflow_tracking: MLflow tracking configuration for experiment logging
    :type mlflow_tracking: MLFlowTracking
    :return: Tuple containing (model_wrapper, extractor_wrapper, feature_test_transformed, y_test)
    :rtype: tuple
    """

    # 1. Load config
    logger.info("Loading configuration...")
    config_path: str = os.environ.get("CONFIG_PATH", file_paths.config_path)
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

        # 5. Set up pipeline for vectorizer and model
        logger.info("Setting up the extractor feature...")
        extractor_wrapper = ExtractorFactory.create_extractor(
            extractor_name=component_sel.extractor_name,
            params=hyperparams.extractor_params
        )
        logger.info("Setting up the model...")
        model_wrapper = ModelFactory.create_model(
            model_name=component_sel.model_name,
            params=hyperparams.model_params
        )

        if training_conf.feature_scaling:
            pipeline = Pipeline([
                ("extractor", extractor_wrapper.vectorizer),
                ("scaler", model_wrapper.scaler),
                ("model", model_wrapper.classifier)
            ], memory=None)  # Optional: Feature scaling
        else:
            pipeline = Pipeline([
                ("extractor", extractor_wrapper.vectorizer),
                ("model", model_wrapper.classifier)
            ], memory=None)

        # 6. Training pipeline strategy with hyperparmeter fine-tuning
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid= hyperparams.param_grid,
            cv=5,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        logger.info(f"The best hyperparameters: {grid_search.best_params_}")

        best_model = grid_search.best_estimator_
        model_wrapper.classifier = best_model.named_steps['model']
        extractor_wrapper.vectorizer = best_model.named_steps['extractor']
        
        logger.info("Transforming test data...")
        feature_test_transformed = extractor_wrapper.vectorizer.transform(X_test)

        # Extract scaler if feature scaling was used
        if training_conf.feature_scaling and 'scaler' in best_model.named_steps:
            logger.info("Applying feature scaling...")
            model_wrapper.scaler = best_model.named_steps['scaler']
            # Apply scaler transformations
            feature_test_transformed = model_wrapper.scaler.transform(feature_test_transformed)
        else:
            # Explicitly mark scaler as unused so downstream code can skip scaling safely.
            model_wrapper.scaler = None

    except Exception as e:
        logger.exception(f"Unexpected error in training pipeline: {e}")
        raise RuntimeError("Training pipeline failed")

    # 7. Evaluate model on test set
    if training_conf.evaluate_after_training:
        logger.info("Evaluating model on test set...")
        evaluate(model_wrapper, feature_test_transformed, y_test)

    # 8. Save model and feature extractor
    # Create new folder with the name 'models' if it doesn't exist
    os.makedirs(config["models"]["dir"], exist_ok=True)

    logger.info("Saving model and extractor...")
    # Dump files
    joblib.dump(model_wrapper, config["models"]["model"])
    joblib.dump(extractor_wrapper, config["models"]["extractor"])

    logger.info("Training pipeline completed")

    return model_wrapper, extractor_wrapper, feature_test_transformed, y_test
