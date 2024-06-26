from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Allow all origins (replace '*' with your frontend URL in a production environment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://127.0.0.1:8000"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/my_model/my_model.keras")
CLASS_NAMES = ['anthracnose', 'healthy']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence) * 100
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)



# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
# import os
#
# app = FastAPI()
#
# # Allow all origins (replace '*' with your frontend URL in a production environment)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://127.0.0.1:5500", "http://127.0.0.1:8000"],  # Add your frontend URLs
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # Load the converted model in SavedModel format
# MODEL = tf.saved_model.load('../saved_models/model.savedmodel1')  # Update the path
#
# CLASS_NAMES = ['anthracnose', 'healthy']
#
#
# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"
#
#
# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image
#
#
#
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     image = read_file_as_image(await file.read())
#     image_batch = np.expand_dims(image, 0)
#
#     # Get the concrete function from the SavedModel
#     infer = MODEL.signatures["serving_default"]
#
#     # Prepare the input tensor
#     input_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)
#
#     # Make predictions
#     predictions = infer(input_tensor)
#
#     # Inspect the keys of the predictions dictionary
#     print("Keys in predictions dictionary:", predictions.keys())
#
#     # Assuming the output key is different, replace "dense" with the correct key
#     output_key = "sequential_9"  # Replace "your_output_key" with the actual output key
#     predicted_class = CLASS_NAMES[np.argmax(predictions[output_key])]
#     confidence = np.max(predictions[output_key])
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence) * 100
#     }
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)



