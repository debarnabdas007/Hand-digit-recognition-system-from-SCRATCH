from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import our router and logger
from app.api.predict import router as predict_router
from app.core.logger import get_logger

logger = get_logger(__name__)

# 1. Ignite the API
app = FastAPI(
    title="Edge Digit Vision API",
    description="Professional ML Backend for PyTorch Digit Recognition",
    version="1.0.0"
)

# 2. Setup CORS (Security policy to allow Streamlit to communicate with us)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you would restrict this to your Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Attach the Receptionist to the Factory
app.include_router(predict_router, prefix="/api/v1")

# 4. A simple health-check endpoint
@app.get("/")
async def health_check():
    logger.info("Health check endpoint pinged.")
    return {"status": "Online", "message": "Edge Digit Vision API is running. Visit /docs to test."}