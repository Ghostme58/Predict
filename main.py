import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load('model/model.pkl')

class Form(BaseModel):
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: int  # 0: mobile, 1: desktop, 2: tablet
    device_os: str
    device_brand: str
    device_model: str
    device_browser: str
    geo_country: str
    geo_city: str
    month: int
    day: int
    hour: int
    day_of_week: int
    is_weekend: int
    width: int
    height: int
    time_period: str

class Prediction(BaseModel):
    Result: int

@app.get('/status')
def status():
    return "I'm OK"

@app.get('/version')
def version():
    return model['metadata']

@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)
    return {
        'Result': int(y[0])
    }