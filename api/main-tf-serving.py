from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
from fastapi.middleware.cors import CORSMiddleware
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TensorFlow Serving endpoint
endpoint = "http://localhost:8501/v1/models/potato_model:predict"

# Class names
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    """
    Read an image file and convert it to a NumPy array.
    """
    try:
        image = Image.open(BytesIO(data))
        image = image.resize((256, 256))  # Resize to match model input shape
        image = np.array(image)
        return image
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)  
):
    """
    Predict the class of an uploaded image.
    """
    try:
        # Read and preprocess the image
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, axis=0)  # Add batch dimension

        # Prepare the payload for TensorFlow Serving
        json_data = {
            "instances": img_batch.tolist()  # Convert to list for JSON serialization
        }

        # Send the request to TensorFlow Serving
        response = requests.post(endpoint, json=json_data)
        response.raise_for_status()  # Raise an error for bad responses

        # Extract predictions
        predictions = response.json().get("predictions", response.json().get("outputs"))
        if predictions is None:
            raise ValueError("No predictions found in the response.")
        
        prediction = np.array(predictions[0])
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return {
            "class": predicted_class,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

    '''docker run -t --rm -p 8501:8501 -v C:/Users/abass/OneDrive/Desktop/cocoa_disease/saved_models:/models
 tensorflow/serving --rest_api_port=8501 --model_name=potato_model --model_base_path=/models'''