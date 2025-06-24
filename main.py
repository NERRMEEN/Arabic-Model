from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model, tokenizer, and label encoder
MODEL_PATH = "arabic_nlp_optimized.keras"
TOKENIZER_PATH = "tokenizer_optimized.pickle"
ENCODER_PATH = "label_encoder_optimized.pickle"
MAX_LEN = 150

model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

app = FastAPI(title="Arabic NLP Model API")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Preprocess: tokenize and pad
    seq = tokenizer.texts_to_sequences([req.text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    # Predict
    preds = model.predict(padded)
    if preds.shape[1] == 1:  # Binary
        confidence = float(preds[0][0])
        label_idx = int(confidence > 0.5)
    else:  # Multi-class
        label_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
    predicted_class = encoder.inverse_transform([label_idx])[0]
    return PredictResponse(predicted_class=predicted_class, confidence=confidence)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)