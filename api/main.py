from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
from database.queries import save_prediction, get_predictions

BASE_DIR = Path(__file__).resolve().parent.parent
PREPROCESSOR_PATH = (BASE_DIR / "models" / "preprocessor.pkl")
MODEL_PATH = (BASE_DIR / "models" / "logistic_regression_tuned.pkl")
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float



@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/predict")
def predict_churn(customer: CustomerData):
    customer_data = customer.model_dump()    
    input_data = pd.DataFrame([customer_data])  
    input_processed = preprocessor.transform(
        input_data
    )
    
    prediction = model.predict(
        input_processed
    )[0]

    probability = model.predict_proba(
        input_processed
    )[0][1]

    probability_percent = probability * 100   
    
    # save prediction ti MYSQL
    save_prediction(
        gender=customer.gender,
        tenure=customer.tenure,
        contract=customer.Contract,
        monthly_charges=customer.MonthlyCharges,
        total_charges=customer.TotalCharges,
        prediction=int(prediction),
        churn_probability=round(
            probability_percent, 2
        )
    )
    
    
    # create result 
    if prediction == 1:
        result = "Customer is likely to Churn"
    else:
        result = "Customer is likely to Stay"

    return {
        "prediction": int(prediction),
        "result": result,
        "churn_probability": round(
            probability_percent,
            2
        )
    }
    
@app.get("/predictions")
def prediction_history():
    predictions=get_predictions()
    return{
        "predictions":predictions
    }    