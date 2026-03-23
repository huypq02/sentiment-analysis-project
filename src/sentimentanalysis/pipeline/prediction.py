from .sentiment_pipeline import SentimentPipeline
from sentimentanalysis.utils import setup_logging


logger = setup_logging(__name__)


def predict(pipeline, feature):
    """
    Make a prediction.
    
    :param pipeline: Sentiment pipeline containing extractor and model
    :type pipeline: SentimentPipeline
    :param feature: Model-ready feature(s) to predict sentiment for
    :type feature: array-like
    :return: Prediction result
    :rtype: array-like
    """

    try:
        # Make a prediction a review text from the customer
        logger.info("Predicting ...")
        return pipeline.predict(feature)

    except Exception as e:
        logger.exception(f"Unexpected error in prediction pipeline: {e}")
        raise RuntimeError("Prediction failed")
