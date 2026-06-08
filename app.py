import pandas as pd 
import joblib
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#Load the saved models files 
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

#App title
st.title("Titanic Model")
st.markdown("Titanic model project to predict the fare.")

with st.sidebar:
    st.header("Passenger Inputs")
    survived = st.number_input("Survived (0/1)", 0, 1)
    pclass = st.number_input("Passenger Class", 1, 3)
    Sibsp = st.number_input("Siblings/Spouses", 0 , 5)
    parch = st.number_input("Parents/Children", 0 , 5)

#Prepare input 
input_data = np.array([[survived , pclass , Sibsp , parch]])

#Scale input 
scale_input = scaler.transform(input_data)

#Predictions
predictions = model.predict(scale_input)

if st.button("Predict Fare"):
    st.success(f"Prediction Value (Fare): {predictions}")

#Passenger info
st.info(f"Passenger Info : Survived = { survived } , PClass = {pclass} , SibSp = {Sibsp} , Parch = {parch}")