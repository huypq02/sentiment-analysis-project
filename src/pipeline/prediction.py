import pandas as pd
import joblib
import os
from .training import train
from src.utils import load_config, setup_logging
from src.data import Preprocessor
from src.config import DEFAULT_CONFIG_PATH

logger = setup_logging(__name__)


def predict(model, extractor, text: str):
    """Make a prediction.
    
    Args:
        model: Trained model
        extractor: Feature extractor
        text: Text to predict sentiment for
        
    Returns:
        prediction (int)
    """

    try:
        # 8. Preprocessing for the review text
        logger.info("Data preprocessing ...")
        preprocessor = Preprocessor()
        tokens = preprocessor.preprocess(text)
        text_clean = " ".join(tokens)
        test_data = pd.Series([text_clean])
        transformed_data = extractor.transform(test_data)

        # 9. Make a prediction a review text from the customer
        logger.info("Predicting ...")
        predictions = model.predict(transformed_data)
        logger.info(f"Predictions: {predictions[0]}")

    except Exception as e:
        logger.exception(f"Unexpected error in prediction pipeline: {e}")
        return RuntimeError("Prediction failed")

    return predictions[0]


if __name__ == "__main__":
    from src.config import (
        DataParameters, 
        ComponentSelection,
        Hyperparameters,
        TrainingConfiguration,
        FilePaths,
        MLFlowTracking
    )

    config_path = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)
    config = load_config(config_path)
    model_path = config["models"]["model"]
    extractor_path = config["models"]["extractor"]

    if os.path.exists(model_path) and os.path.exists(extractor_path):
        model = joblib.load(config["models"]["model"])
        extractor = joblib.load(config["models"]["extractor"])
    else:
        model, extractor, _, _, _ = train(
            data_params=DataParameters(),
            component_sel=ComponentSelection(),
            hyperparams=Hyperparameters(),
            training_conf=TrainingConfiguration(),
            file_paths=FilePaths(),
            mlflow_tracking=MLFlowTracking()
        )

    predict(
        model=model,
        extractor=extractor,
        text="The books are really not bad!"
    )
