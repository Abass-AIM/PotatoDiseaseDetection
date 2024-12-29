from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import tensorflow as tf

app = FastAPI()

# TensorFlow Serving REST API endpoint
endpoint = "http://localhost:8001/v1/potato-disease/potato_model:predict"

# Class names for predictions
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).resize((256,256))
    return np.array(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)
                  ):
    
    # Read and preprocess the uploaded file
    image_data = await file.read()
    image = read_file_as_image(image_data)
    img_batch = np.expand_dims(image, 0)  # Add batch dimension

    # Prepare data for TensorFlow Serving
    json_data = {"instances": img_batch.tolist()}

    # Make a POST request to TensorFlow Serving
    response = requests.post(endpoint, json=json_data)
    

    # Parse the response and extract predictions
    prediction = np.array(response.json()["predictions"][0])
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence":(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)
       
#docker run -p 8000:8501 -v C:/Users/abass/OneDrive/Desktop/potato-disease:/potato-disease tensorflow/serving --rest_api_port=8501 --model_config_file=/potato-disease/models.config.txt