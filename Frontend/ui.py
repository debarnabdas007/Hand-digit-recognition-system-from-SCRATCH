import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests
import base64
from PIL import Image
import io
import numpy as np
import os

# Dynamically grab the URL from Docker, or default to localhost if running manually
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api/v1/predict")

st.set_page_config(page_title="Digit Vision UI", layout="centered")

st.title("Edge-Optimized Digit Vision")
st.markdown("Draw a **single digit (0-9)** in the box below. The FastAPI microservice will process it.")

# 1. Create the interactive drawing canvas
canvas_result = st_canvas(
    fill_color="black",      
    stroke_width=15,         # Thick pen for 28x28 downscaling
    stroke_color="white",    # MNIST format is white text on black background
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 2. Add a button to manually trigger the API call
if st.button("Predict Digit"):
    if canvas_result.image_data is not None:
        
        # Check if the canvas is completely blank (all black pixels)
        if np.any(canvas_result.image_data[:, :, 0] > 0):
            with st.spinner("Sending to FastAPI Backend..."):
                try:
                    # Step A: Convert the canvas pixel array to a PNG image
                    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                    
                    # Step B: Convert the PNG image to a Base64 string
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    # Step C: Package it into the exact JSON our Pydantic schema expects
                    payload = {"image_data": base64_string}
                    
                    # Step D: Fire the POST request across the network
                    response = requests.post(API_URL, json=payload)
                    
                    # Step E: Read the Receptionist's reply
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success(f"### PyTorch Predicts: {result['prediction']}")
                        st.info(f"**Confidence Score:** {result['confidence']}%")
                    else:
                        st.error(f"Backend rejected the request: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the backend. Is FastAPI running on port 8000?")
        else:
            st.warning("Please draw a digit on the canvas first!")