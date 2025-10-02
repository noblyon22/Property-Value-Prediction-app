import streamlit as st
import pandas as pd
import joblib

# Load trained model
with open("XGBoost.pkl", "rb") as file:
    model = joblib.load(file)

st.title("Property Value Prediction App")
st.write("Enter the property details below to predict the value.")

# Input fields with default values
CRIM = st.number_input("Crime rate", min_value=0.0, value=0.1, step=0.01)
ZN = st.number_input("Proportion of residential land zoned", min_value=0.0, value=12.5, step=0.1)
INDUS = st.number_input("Proportion of non-retail business acres", min_value=0.0, value=7.0, step=0.1)
NOX = st.number_input("Nitric oxides concentration", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
RM = st.number_input("Average number of rooms per dwelling", min_value=1.0, max_value=10.0, value=6.0, step=0.1)
AGE = st.number_input("Proportion of units built before 1940", min_value=0.0, value=50.0, step=1.0)
DIS = st.number_input("Weighted distances to employment centers", min_value=0.0, value=5.0, step=0.01)
RAD = st.number_input("Index of accessibility to radial highways", min_value=0.0, value=4.0, step=1.0)
TAX = st.number_input("Full-value property tax rate", min_value=0.0, value=300.0, step=1.0)
PTRATIO = st.number_input("Pupil-teacher ratio", min_value=0.0, value=15.0, step=0.1)
B = st.number_input("Proportion of African American population", min_value=0.0, value=350.0, step=1.0)
LSTAT = st.number_input("Lower status population percentage", min_value=0.0, value=12.0, step=0.1)

if st.button("Predict"):
    try:
        input_data = pd.DataFrame([[CRIM, ZN, INDUS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]],
                                  columns=["CRIM","ZN","INDUS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"])
        prediction = model.predict(input_data)
        st.success(f"🏠 Predicted Property Value: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
