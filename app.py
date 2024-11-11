import requests
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import joblib

# Google Drive file ID for your species.pkl file
GDRIVE_FILE_ID = "1B9Zpi4m59LI2BQC2SRb2yR1UyiI72NRx"
GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
PICKLE_FILE_PATH = "species_model.pkl"

# Function to download the pickle file
def download_pickle_file():
    response = requests.get(GDRIVE_URL)
    if response.status_code == 200:
        with open(PICKLE_FILE_PATH, "wb") as f:
            f.write(response.content)
        print("Pickle file downloaded successfully.")
    else:
        print("Failed to download the pickle file.")

# Call this function to download the pickle file before the API starts
download_pickle_file()

# Load the pre-trained model and encoder
species_model = joblib.load('species_model.pkl')  # Your SVC model
species_encoder = joblib.load('species_encoder.pkl')  # Your LabelEncoder

app = FastAPI()

# Function to process image and prepare for prediction
def process_image(image: Image.Image):
    img = image.convert('RGB')  # Convert to RGB
    img = img.resize((128, 128))  # Resize image to match model input size
    img_array = np.array(img)
    img_array_flattened = img_array.flatten().reshape(1, -1)  # Flatten image
    return img_array_flattened

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image file
    img = Image.open(file.file)
    
    # Process image for prediction
    img_array_flattened = process_image(img)
    
    # Make prediction
    prediction = species_model.predict(img_array_flattened)
    
    # Decode the prediction to get the species name
    predicted_label = species_encoder.inverse_transform(prediction)
    
    return {"predicted_species": predicted_label[0]}
