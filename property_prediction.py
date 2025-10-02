# property_prediction.py
import streamlit as st
import pandas as pd
import joblib
import xgboost
from sklearn.preprocessing import StandardScaler  # only if used in training

# Load trained model
try:
    with open("XGBoost.pkl", "rb") as file:
        model = joblib.load(file)
except FileNotFoundError:
    st.error("Model file 'XGBoost.pkl' not found. Please check the file path.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# App title
st.title("🏠 Property Value Prediction App")
st.write("Enter the property details below to predict the value.")

# Input fields
crim = st.number_input("Crime rate", min_value=0.0, value=0.1, step=0.01)
zn = st.number_input("Proportion of residential land zoned", min_value=0.0, value=12.5, step=0.1)
indus = st.number_input("Proportion of non-retail business acres", min_value=0.0, value=7.0, step=0.1)
chas = st.selectbox("Bounds Charles River? (1 = yes, 0 = no)", options=[0, 1])
nox = st.number_input("Nitric oxides concentration", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
rm = st.number_input("Average number of rooms per dwelling", min_value=1.0, max_value=10.0, value=6.0, step=0.1)
age = st.number_input("Proportion of units built before 1940", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
dis = st.number_input("Weighted distances to employment centers", min_value=0.0, value=5.0, step=0.01)
rad = st.number_input("Index of accessibility to radial highways", min_value=1, max_value=24, value=4, step=1)
tax = st.number_input("Full-value property tax rate", min_value=0.0, value=300.0, step=1.0)
ptratio = st.number_input("Pupil-teacher ratio", min_value=0.0, value=15.0, step=0.1)
b = st.number_input("Proportion of African American population", min_value=0.0, max_value=400.0, value=350.0, step=1.0)
lstat = st.number_input("Lower status population percentage", min_value=0.0, max_value=100.0, value=12.0, step=0.1)

# Predict button
if st.button("Predict"):
    try:
        # Put inputs into DataFrame
        input_data = pd.DataFrame([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]],
                                  columns=['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat'])
        # Predict
        prediction = model.predict(input_data)
        st.success(f"🏠 Predicted Property Value: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
