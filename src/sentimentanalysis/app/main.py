from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, HTTPException
import uvicorn
import os
import joblib
from sentimentanalysis.pipeline import predict
from sentimentanalysis.utils import load_config
from sentimentanalysis.config import (
    SERVICE_NAME,
    API_VERSION,
    DEFAULT_CONFIG_PATH,
    HEALTHY_STATUS
)
from .schemas import ReviewRequest, ReviewResponse

model = None
extractor = None
config = None

def load_artifact():
    """
    Load trained model and feature extractor from disk.
    Raises an error if models do not exist (training should be run separately).
    
    :return: None
    :rtype: None
    :raises FileNotFoundError: If trained models are not found
    """
    global model, extractor, config

    config_path = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH) 
    config = load_config(config_path)
    model_path = config["models"]["model"]
    extractor_path = config["models"]["extractor"]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first: python -m sentimentanalysis.pipeline.training")
    if not os.path.exists(extractor_path):
        raise FileNotFoundError(f"Extractor not found at {extractor_path}. Run training first: python -m sentimentanalysis.pipeline.training")
    
    model = joblib.load(model_path)
    extractor = joblib.load(extractor_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    :param app: FastAPI application instance
    :type app: FastAPI
    :return: Async context manager
    :rtype: AsyncGenerator
    """
    load_artifact()
    yield

app = FastAPI(lifespan=lifespan)
router = APIRouter(prefix="/v1")

@router.get("/health")
def health_check():
    """
    Health check endpoint to verify service status.
    
    :return: Service health status information
    :rtype: dict
    """
    return {
        "status": HEALTHY_STATUS,
        "service": SERVICE_NAME,
        "version": API_VERSION,
    }


@router.post("/predictions")
async def prediction(request: ReviewRequest):
    """
    Predict sentiment for the provided review text.
    
    :param request: Review request containing text to analyze
    :type request: ReviewRequest
    :return: Prediction response sentiment
    :rtype: ReviewResponse
    :raises HTTPException: If text is empty or None
    """
    text = request.text
    if text is None or text == "":
        raise HTTPException(status_code=400, detail="The text should not be empty or none.")

    sentiment = predict(
        model=model,
        extractor=extractor,
        text=text
    )

    return ReviewResponse(text=text, sentiment=sentiment)

app.include_router(router)


if __name__ == "__main__":
    config = uvicorn.Config("sentimentanalysis.app.main:app", port=8080, log_level="info")
    server = uvicorn.Server(config)
    server.run()
