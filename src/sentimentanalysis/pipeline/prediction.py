import pandas as pd
import joblib
import os
from .training import train
from sentimentanalysis.utils import load_config, setup_logging
from sentimentanalysis.data import Preprocessor
from sentimentanalysis.config import DEFAULT_CONFIG_PATH

logger = setup_logging(__name__)


def predict(model, extractor, text: str):
    """
    Make a prediction.
    
    :param model: Trained model
    :type model: SentimentModel
    :param extractor: Feature extractor
    :type extractor: BaseFeatureExtractor
    :param text: Text to predict sentiment for
    :type text: str
    :return: Prediction result
    :rtype: int
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
