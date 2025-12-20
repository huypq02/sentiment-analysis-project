import pandas as pd
from .train_pipeline import train_main
from src.utils.logger import setup_logging

logger = setup_logging(__name__)

def predict_main(model=None,  text=None, feature_test_scaled=None):
    """Make a prediction."""

    try:
        if model is None or feature_test_scaled is None:
            # 1-7 Implement the model
            model, extractor, feature_test_scaled, _, _ = train_main()

        # 8. Make a prediction
        test_data = pd.Series([text])
        predictions = model.predict(extractor.transform(test_data))

        print("predictions:::", predictions)

    except Exception as e:
        logger.exception(f'Unexpected error in prediction pipeline: {e}')
        return None

    return predictions


if __name__ == "__main__":
    predict_main()