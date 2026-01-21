import pandas as pd
from .training import train
from src.utils import setup_logging
from src.data import Preprocessor

logger = setup_logging(__name__)


def predict(model=None, text=None, feature_test_scaled=None):
    """Make a prediction.
    
    Args:
        model: Trained model (optional, will load if None)
        text: Text to predict sentiment for
        feature_test_scaled: Pre-transformed features (optional)
        
    Returns:
        prediction (int)
    """

    try:
        if model is None or feature_test_scaled is None:
            # 1-7 Implement the model
            model, extractor, feature_test_scaled, _, _ = train()

        # 8. Preprocessing for the review text
        logger.info("Data preprocessing ...")
        preprocessor = Preprocessor()
        tokens = preprocessor.preprocess(text=text)
        text_clean = " ".join(tokens)

        # 9. Make a prediction a review text from the customer
        test_data = pd.Series([text_clean])
        transformed_data = extractor.transform(test_data)
        predictions = model.predict(transformed_data)
        logger.info(f"Predictions: {predictions[0]}")

    except Exception as e:
        logger.exception(f"Unexpected error in prediction pipeline: {e}")
        return None

    return predictions[0]


if __name__ == "__main__":
    predict()
