from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import uvicorn
import os
import joblib
from sentimentanalysis.pipeline import predict, train
from sentimentanalysis.utils import load_config
from sentimentanalysis.config import (
    DataParameters, 
    ComponentSelection,
    Hyperparameters,
    TrainingConfiguration,
    FilePaths,
    MLFlowTracking,
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
    Load trained model and feature extractor from disk or train new ones if not found.
    
    :return: None
    :rtype: None
    """
    global model, extractor, config

    config_path = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH) 
    config = load_config(config_path)
    model_path = config["models"]["model"]
    extractor_path = config["models"]["extractor"]
    if os.path.exists(model_path) and os.path.exists(extractor_path):
        model = joblib.load(model_path)
        extractor = joblib.load(extractor_path)
    else:
        model, extractor, _, _, _ = train(
            data_params=DataParameters(),
            component_sel=ComponentSelection(),
            hyperparams=Hyperparameters(),
            training_conf=TrainingConfiguration(),
            file_paths=FilePaths(),
            mlflow_tracking=MLFlowTracking()
        )

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


@app.get("/health")
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


@app.post("/predictions")
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


if __name__ == "__main__":
    config = uvicorn.Config("sentimentanalysis.app.main:app", port=8080, log_level="info")
    server = uvicorn.Server(config)
    server.run()
