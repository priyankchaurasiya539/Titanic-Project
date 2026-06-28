import pandas as pd 
import joblib
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel , Field 
from typing import  Literal , Annotated

#Import the dataset
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

app = FastAPI()
#build a pydantic model to validate incoming data

class PassengerInput(BaseModel):
    Pclass : Annotated[Literal[1 , 2 , 3] , Field(... , description="Passenger Class")]
    # Age : Annotated[int , Field(... , description="Age of the passenger"  , strict=True , ge=0 , lt=100)]
    SibSp : Annotated[int ,Field(...  , description="Siblings and Spouse" , strict= True ,  ge=0 , le=8) ]
    Parch : Annotated[int ,Field(...  , description="Parent and child" , strict= True , ge=0 , le=6)]


@app.post("/predict")
def predict_input(data : PassengerInput):
    input_df = pd.DataFrame({
        "Pclass" : [data.Pclass],
        # "Age" : [data.Age],
        "SibSp" : [data.SibSp],
        "Parch" : [data.Parch]
    })

    prediction = model.predict(input_df)

    return JSONResponse(status_code=200 , content={"Predicted Fare" : round(float(prediction[0]) , 2)})
