from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import Annotated
import joblib,os
import pandas as pd
from config import MODEL_DIR
from model.data_sourcing import Data_Sourcing

import warnings
warnings.filterwarnings(action='ignore')

# defining the pydantic schema
class EPP_Features(BaseModel):
    CREDIT_LIMIT: Annotated[float,Field(...,description='Credit Limit of the customer',gt=0,example=100.00)]
    BALANCE: Annotated[float,Field(...,description='Card balance of the customer',ge=0,example=20.00)]  
    PURCHASES: Annotated[float,Field(...,description='Amount of customer purchases',ge=0,example=50.00)]
    CASH_ADVANCE: Annotated[float,Field(...,description='Amount of customers cash advance',ge=0,example=50.00)]
    PAYMENTS: Annotated[float,Field(...,description='Payment amount of customer',ge=0,example=50.00)]
    TENURE: Annotated[int,Field(...,description='Tenure of customer in months',ge=0,example=12)]                    

    @validator("BALANCE")
    def balance_cannot_exceed_credit(v, values):
        cred_lim = values.get("CREDIT_LIMIT")
        if cred_lim is not None and v > cred_lim:
            raise ValueError("Balance cannot exceed Credit Limit")
        return v

# load the model
model=joblib.load(os.path.join(MODEL_DIR,"ensemble_model.pkl"))
# load the scaler
scaler=joblib.load(os.path.join(MODEL_DIR,"scaler.pkl"))
# load the feature names and order
feature_order=joblib.load(os.path.join(MODEL_DIR,"feature_names.pkl")).tolist()

# creating an instance of the FastAPI class
app=FastAPI(title="EPP Predictor",description="The API predicts if the customer opts for EPP: Easy Payment Plan or not")

# specifying that our HTML templates are stored in a folder called templates
templates=Jinja2Templates(directory='templates')
# mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/',response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse('home.html',{'request':request,'result':None})

@app.api_route('/predict', methods=['GET','POST'],response_class=HTMLResponse)
async def predict(
    request: Request,
    CREDIT_LIMIT: float=Form(...),
    BALANCE: float=Form(...),
    PURCHASES: float=Form(...),
    CASH_ADVANCE: float=Form(...),
    PAYMENTS: float=Form(...),
    TENURE: int=Form(...)

):
    UTILIZATION=BALANCE/CREDIT_LIMIT if CREDIT_LIMIT > 0 else 0
    # form data
    data={'CREDIT_LIMIT':CREDIT_LIMIT,'BALANCE':BALANCE,'PURCHASES':PURCHASES,'CASH_ADVANCE':CASH_ADVANCE,'PAYMENTS':PAYMENTS,'TENURE':TENURE,'UTILIZATION':UTILIZATION}
    input_df=pd.DataFrame([data])
    # retain the same column order as training
    input_df=input_df[feature_order]
    # scale features
    scaled_df=scaler.transform(input_df)
    # make prediction
    prediction=model.predict(scaled_df)[0]
    prediction_prob=model.predict_proba(scaled_df)[0][1]
    # limiting decimals conditionally
    prediction_prob=Data_Sourcing().format_probability(prob=prediction_prob)
    # result="Customer opts for EPP" if prediction==1 else "Customer does not opt for EPP"
    result = "Customer opts for EPP 😊" if prediction==1 else "Customer does not opt for EPP 😞"
    # return the prediction
    return templates.TemplateResponse(
        'home.html',
        {
            'request':request,
            'result': result,
            'probability': f"{prediction_prob}",
            'UTILIZATION': f"{round(UTILIZATION,2)}%"
        }
    )