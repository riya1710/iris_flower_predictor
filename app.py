# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('iris_model.pkl')

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")
st.title("🌸 Iris Flower Species Predictor")

st.write("Enter the features of the flower:")

# User input
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prepare input
input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                        columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"🌼 Predicted Species: **{species[prediction]}**")
