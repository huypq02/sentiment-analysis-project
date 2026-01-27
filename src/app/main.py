from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import uvicorn
import os
import joblib
from src.pipeline import predict, train
from src.utils import rating_to_sentiment, load_config
from src.config import (
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
    load_artifact()
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check():
    return {
        "status": HEALTHY_STATUS,
        "service": SERVICE_NAME,
        "version": API_VERSION,
    }


@app.post("/predict")
async def prediction(request: ReviewRequest):
    text = request.text
    if text is None or text == "":
        raise HTTPException(status_code=400, detail="The text should not be empty or none.")

    rating = predict(
        model=model,
        extractor=extractor,
        text=text
    )
    sentiment = rating_to_sentiment(rating)

    return ReviewResponse(text=text, rating=rating, sentiment=sentiment)


if __name__ == "__main__":
    config = uvicorn.Config("src.app.main:app", port=8080, log_level="info")
    server = uvicorn.Server(config)
    server.run()
