import streamlit as st
import pickle as pk
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from streamlit_lottie import st_lottie
import json

df = pd.read_excel("CarPrice.xlsx")

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
lottie_animation = load_lottiefile("car.json")

st.title("Car Price Prediction")
st_lottie(lottie_animation, height=300, key="car_animation")
le_brand = LabelEncoder()
le_model = LabelEncoder()
le_fuel = LabelEncoder()
le_transmission = LabelEncoder()
le_owner = LabelEncoder()
df['Brand'] = le_brand.fit_transform(df['Brand'])
df['Model'] = le_model.fit_transform(df['Model'])
df['Fuel'] = le_fuel.fit_transform(df['Fuel'])
df['Transmission'] = le_transmission.fit_transform(df['Transmission'])
df['Owner'] = le_owner.fit_transform(df['Owner'])

brand = st.selectbox("Select Brand", le_brand.classes_)
model = st.selectbox("Select Model", le_model.classes_)
fuel = st.selectbox("Select Fuel Type", le_fuel.classes_)
transmission = st.selectbox("Select Transmission Type", le_transmission.classes_)
owner = st.selectbox("Select Owner Type", le_owner.classes_)

year = st.number_input("Year of the car:", min_value=1900, max_value=2025,value=1992)
km_driven = st.number_input("Kilometers Driven:", min_value=0, max_value=1000000 ,value=500000)

brand_encoded = le_brand.transform([brand])[0]
model_encoded = le_model.transform([model])[0]
fuel_encoded = le_fuel.transform([fuel])[0]
transmission_encoded = le_transmission.transform([transmission])[0]
owner_encoded = le_owner.transform([owner])[0]

input_data = [[brand_encoded, model_encoded, year, km_driven, fuel_encoded, transmission_encoded, owner_encoded]]

filename = 'xgboost_model.sav'
loaded_model = pk.load(open(filename, 'rb'))

if st.button("Predict"):
    prediction = loaded_model.predict(input_data)
    st.success(f"Predicted Selling Price: {prediction[0]:.2f}")
