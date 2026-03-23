import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
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
from .sentiment_pipeline import SentimentPipeline

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
    logger.debug("Using config path: %s", config_path)
    logger.debug("Resolved training data path: %s", data_path)

    sentiment_pipeline = SentimentPipeline()

    try:
        # 2. Load data
        logger.info("Loading data...")
        df = sentiment_pipeline.load_dataset(data_path)

        # 3. Preprocessing
        logger.info("Data preprocessing...")
        texts_cleaned, labels = sentiment_pipeline.extract_texts_and_labels(
            df, data_params.text_column, data_params.label_column
        )
        logger.info(f"Sentiment distribution: {labels.value_counts().to_dict()}")
        logger.debug("Using text column '%s' and label column '%s'", data_params.text_column, data_params.label_column)

        # 4. Split
        logger.info("Splitting dataset...")
        X_train, X_test, y_train, y_test = sentiment_pipeline.split_data(
            texts_cleaned, labels, training_conf.test_size, training_conf.random_state
        )
        logger.debug("Train/Test sizes: %d/%d", len(X_train), len(X_test))

        # 5. Set up extractor and model
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
        logger.debug("Extractor selected: %s", component_sel.extractor_name)
        logger.debug("Model selected: %s", component_sel.model_name)

        if training_conf.feature_scaling:
            sklearn_pipeline = Pipeline([
                ("extractor", extractor_wrapper.vectorizer),
                ("scaler", model_wrapper.scaler),
                ("model", model_wrapper.classifier)
            ], memory=None)  # Optional: Feature scaling
        else:
            sklearn_pipeline = Pipeline([
                ("extractor", extractor_wrapper.vectorizer),
                ("model", model_wrapper.classifier)
            ], memory=None)

        # 6. Training with hyperparameter fine-tuning
        grid_search = GridSearchCV(
            estimator=sklearn_pipeline,
            param_grid=hyperparams.param_grid,
            cv=5,
            n_jobs=-1
        )
        logger.debug("Grid search parameter groups: %d", len(hyperparams.param_grid))

        grid_search.fit(X_train, y_train)
        logger.info(f"The best hyperparameters: {grid_search.best_params_}")

        best_model = grid_search.best_estimator_
        model_wrapper.classifier = best_model.named_steps['model']
        extractor_wrapper.vectorizer = best_model.named_steps['extractor']

        # Extract scaler if feature scaling was used
        if training_conf.feature_scaling and 'scaler' in best_model.named_steps:
            model_wrapper.scaler = best_model.named_steps['scaler']
        else:
            # Explicitly mark scaler as unused so downstream code can skip scaling safely.
            model_wrapper.scaler = None

        # Update sentiment pipeline with trained wrappers
        sentiment_pipeline.set_extractor(extractor_wrapper)
        sentiment_pipeline.set_model(model_wrapper)

        # Transform test features through sentiment pipeline
        feature_test_transformed = sentiment_pipeline.transform(X_test)

    except Exception as e:
        logger.exception(f"Unexpected error in training pipeline: {e}")
        raise RuntimeError("Training pipeline failed")

    # 7. Evaluate model on test set
    if training_conf.evaluate_after_training:
        logger.info("Evaluating model on test set...")
        evaluate(sentiment_pipeline, feature_test_transformed, y_test)

    # 8. Save model and feature extractor
    # Create new folder with the name 'models' if it doesn't exist
    os.makedirs(config["models"]["dir"], exist_ok=True)

    logger.info("Saving model and extractor...")
    # Dump files
    joblib.dump(model_wrapper, config["models"]["model"])
    joblib.dump(extractor_wrapper, config["models"]["extractor"])

    logger.info("Training pipeline completed")

    return model_wrapper, extractor_wrapper, feature_test_transformed, y_test
