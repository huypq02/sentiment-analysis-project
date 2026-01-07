import pandas as pd
from .train_pipeline import train_main
from src.utils import setup_logging
from src.data import Preprocessor

logger = setup_logging(__name__)


def predict_main(model=None, text=None, feature_test_scaled=None):
    """Make a prediction."""

    try:
        if model is None or feature_test_scaled is None:
            # 1-7 Implement the model
            model, extractor, feature_test_scaled, _, _ = train_main()

        # 8. Preprocessing for the review text
        logger.info("Data preprocessing ...")
        preprocessor = Preprocessor()
        tokens = preprocessor.preprocess(text=text)
        text_clean = " ".join(tokens)

        # 9. Make a prediction a review text from the customer
        test_data = pd.Series([text_clean])
        predictions = model.predict(extractor.transform(test_data))
        logger.info(f"Predictions: {predictions[0]}")

    except Exception as e:
        logger.exception(f"Unexpected error in prediction pipeline: {e}")
        return None

    return predictions[0]


if __name__ == "__main__":
    predict_main()
