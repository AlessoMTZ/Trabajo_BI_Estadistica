from fastapi import FastAPI, HTTPException
from flask import Flask, request, jsonify
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load('modelo_entrenado.pkl')  
vectorizer = joblib.load('vectorizer.pkl')
tfid = joblib.load('tfid_transformer.pkl')

class Review(BaseModel):
    text: str

@app.post("/predict/")
def predict(review: Review):
    data = request.get_json(force=True)
    review = data['review']
    words = ' '.join(map(str, review.split()))
    transformed = vectorizer.transform([words])
    transformed = tfid.transform(transformed)
    prediction = model.predict(transformed)
    proba = model.predict_proba(transformed)
    return jsonify({'prediction': int(prediction[0]), 'probability': max(proba[0])})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

from pydantic import BaseModel

class Review(BaseModel):
    text: str

