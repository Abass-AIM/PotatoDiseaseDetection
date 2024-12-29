from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.layers import TFSMLayer
from keras import Sequential


app = FastAPI()


# Define the path to the SavedModel
export_dir = (r"C:/Users/abass/OneDrive/Desktop/Potato-disease/saved_models/1")

# Load the model as a TFSMLayer
MODEL = Sequential([
    TFSMLayer(export_dir, call_endpoint="serving_default")
])
#MODEL = tf.keras.models.load_model(r"C:/Users/abass/OneDrive/Desktop/Potato-disease/saved_models/1.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class' : predicted_class,
        'confidence': float(confidence) 
    }
    
    
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8003)


