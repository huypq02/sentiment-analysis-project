import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
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

        pipeline = Pipeline([
            ("extractor", extractor_wrapper.vectorizer),
            ("model", model_wrapper.classifier)
        ])

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
        feature_test_transformed = extractor_wrapper.vectorizer.transform(X_test)
        # if (
        #     training_conf.feature_scaling
        # ):
        #     feature_train_scaled, feature_test_scaled = model.scale_feature(
        #         feature_train, feature_test
        #     )  # Feature scaling
        #     model.train(feature_train_scaled, y_train, parameters)  # Train data on the model
        # else:
        #     model.train(feature_train, y_train, parameters)
        #     feature_test_scaled = feature_test  # Use unscaled features

    except Exception as e:
        logger.exception(f"Unexpected error in training pipeline: {e}")
        raise RuntimeError("Training pipeline failed")

    # 7. Save model and feature extractor
    # Create new folder with the name 'models' if it doesn't exist
    os.makedirs(config["models"]["dir"], exist_ok=True)

    logger.info("Saving model and extractor...")
    # Dump files
    joblib.dump(model_wrapper, config["models"]["model"])
    joblib.dump(extractor_wrapper, config["models"]["extractor"])

    logger.info("Training pipeline completed")

    return model_wrapper, extractor_wrapper, feature_test_transformed, y_test


if __name__ == "__main__":
    train(
        data_params=DataParameters(
            data_path=os.path.join("data", "raw", "book_reviews_10k_tailored.csv")
        ),
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
                "extractor__binary": [False],  # important

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
