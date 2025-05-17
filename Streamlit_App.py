# heart_disease_pipeline.py


import pandas as pd
import joblib
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sklearn 
print(sklearn.__version__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['pulse_pressure'] = X['ap_hi'] - X['ap_lo']
        X['map'] = (X['ap_hi'] + 2 * X['ap_lo']) / 3
        X['bmi'] = X['weight'] / ((X['height'] / 100) ** 2)  # convert height from cm to meters
        X['sys_dsys_ratio'] = X['ap_hi'] / X['ap_lo']
        X['age_box'] = (X['age'] * 365.25) ** 1.5  # convert age from years to days then apply transformation
        return X


def load_model(path="D:/Heart disease project/model.pkl"):
    return joblib.load(path)

def main():
    st.title("Heart Disease Prediction App")

    st.write("Enter patient details below:")

    age = st.number_input("Age (in years)", 18, 100, 50)
    height = st.number_input("Height (in cm)", min_value=100.0, max_value=250.0, step=0.1)
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, step=0.1)
    ap_hi = st.number_input("Systolic BP", min_value=80, max_value=250, step=1)
    ap_lo = st.number_input("Diastolic BP", min_value=40, max_value=150, step=1)
    cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3], format_func=lambda x: f"Level {x}")
    gluc = st.selectbox("Glucose Level", [1, 2, 3], format_func=lambda x: f"Level {x}")
    gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")


    if st.button("Predict"):
        input_data = pd.DataFrame([{
            'age': age*365,
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'gender': gender,
            'cholesterol': cholesterol,
            'gluc': gluc
        }])

        try:
            model = load_model("D:/Heart disease project/model.pkl")
            prediction = model.predict(input_data)[0]
            result = '⚠️ High likelihood of heart disease' if prediction == 1 else '✅ Low likelihood of heart disease'
            st.success(f"Prediction: {result}")
        except FileNotFoundError:
            st.error("Model file not found. Please train and save the model first.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
