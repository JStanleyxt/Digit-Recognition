from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np

# Use the exact same model structure as training
model = nn.Linear(28 * 28, 10)
model.load_state_dict(torch.load("digit_model.pth", map_location=torch.device("cpu")))
model.eval()

app = FastAPI(title="Digit Recognition API")

@app.get("/")
def health_check():
    return {"message": "Digit Recognition API is running"}

# Input data schema
class ImageInput(BaseModel):
    pixels: list  # List of 784 floats (28x28 image flattened)

@app.post("/predict")
def predict(input: ImageInput):
    if len(input.pixels) != 784:
        return {"error": "Expected 784 pixels (28x28 image)"}

    # Convert to tensor
    x = torch.tensor(input.pixels, dtype=torch.float32).view(1, -1)
    with torch.no_grad():
        output = model(x)
        prediction = torch.argmax(output, dim=1).item()

    return {"predicted_digit": prediction}
