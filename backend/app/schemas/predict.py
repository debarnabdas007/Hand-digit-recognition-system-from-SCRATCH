from pydantic import BaseModel, Field

# What the API expects to RECEIVE from Streamlit
class PredictRequest(BaseModel):
    image_data: str = Field(
        ..., 
        description="A Base64 encoded string representing the user's drawn image."
    )

# What the API promises to RETURN to Streamlit
class PredictResponse(BaseModel):
    prediction: int = Field(
        ..., 
        description="The final predicted digit from 0 to 9."
    )
    confidence: float = Field(
        ..., 
        description="The confidence percentage of the prediction."
    )


