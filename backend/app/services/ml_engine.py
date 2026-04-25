import base64
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Import our custom rulebook
from app.core.logger import get_logger
from app.core.exceptions import ModelLoadError, InvalidImageError

logger = get_logger(__name__)

# The exact physical NN structure from notebook
class MNISTVisionModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7, out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

# The Engine that runs the math and holds the model in memory
class MLEngine:
    def __init__(self, model_path: str = "models/edge_digit_vision_final.pth"):
        # Defaulting to CPU is standard for web servers (cheaper hosting) unless GPU is explicitly available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing ML Engine on device: {self.device}")
        
        try:
            
            self.model = MNISTVisionModel(input_shape=1, hidden_units=32, output_shape=10).to(self.device)
            
            # Load the memories (.pth file). map_location ensures it loads safely even on CPU-only servers
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            # CRITICAL: Lock the model so weights don't update during user requests
            self.model.eval() 
            logger.info("PyTorch model loaded successfully.")
            
            # Setup the exact math transformation used during Colab training
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
        except FileNotFoundError:
            logger.error(f"Model file not found at {model_path}. Did you put it in the models/ folder?")
            raise ModelLoadError(detail="Model artifact missing from server.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelLoadError()

    def predict(self, base64_string: str) -> dict:
        """
        Takes a base64 image string from Streamlit, preprocesses it, and runs inference.
        """
        try:
            # 1. Strip the HTML canvas header if Streamlit sends it (e.g., "data:image/png;base64,...")
            if "," in base64_string:
                base64_string = base64_string.split(",")[1]
                
            # 2. Convert text string back into raw image bytes
            image_bytes = base64.b64decode(base64_string)
            
            # 3. Open as an image, convert to Grayscale ('L'), and resize to exactly 28x28
            img = Image.open(io.BytesIO(image_bytes)).convert('L').resize((28, 28))
            
            # 4. Turn into a math tensor and add batch dimension -> Shape becomes [1, 1, 28, 28]
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise InvalidImageError()

        # 5. Run the Math
        with torch.inference_mode():
            logits = self.model(img_tensor)
            prediction = logits.argmax(dim=1).item()
            
            # Calculate confidence % using softmax
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence = probabilities[prediction].item() * 100
            
            logger.info(f"Prediction successful: Digit {prediction} with {confidence:.2f}% confidence.")
            
            return {
                "prediction": prediction,
                "confidence": round(confidence, 2)
            }