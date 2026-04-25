from fastapi import APIRouter, Depends
from functools import lru_cache

# Import  schemas, engine, and logger
from app.schemas.predict import PredictRequest, PredictResponse
from app.services.ml_engine import MLEngine
from app.core.logger import get_logger

logger = get_logger(__name__)

# Create the router 
router = APIRouter()

# Dependency Injection: Load the PyTorch engine once and keep it in RAM
@lru_cache()
def get_ml_engine():
    return MLEngine()

#  API endpoint 
@router.post("/predict", response_model=PredictResponse)
async def predict_digit(request: PredictRequest, engine: MLEngine = Depends(get_ml_engine)):
    logger.info("Received new prediction request from frontend.")
    
    # Hand the validated Base64 string to the Engine Room
    result = engine.predict(request.image_data)
    
    # Package the result using our strict outgoing Pydantic schema
    return PredictResponse(
        prediction=result["prediction"], 
        confidence=result["confidence"]
    )