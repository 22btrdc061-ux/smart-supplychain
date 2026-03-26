import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Smart Grocery Shelf-Life Predictor")

# Load model
model = joblib.load("rf_model.pkl")

# Load dataset
df = pd.read_csv("final_project_dataset.csv")

# Select item
item_list = df['item'].unique()
item = st.selectbox("Select Grocery Item", item_list)

# Inputs
temp = st.slider("Temperature (°C)", 0, 50, 30)
humidity = st.slider("Humidity (%)", 0, 100, 60)

# Get item data safely
item_data = df[df['item'] == item]

if item_data.empty:
    st.error("Item not found")
else:
    item_data = item_data.iloc[0]

    ideal_temp = item_data['ideal_temp_c']
    q10 = item_data['q10']

    # Feature engineering
    temp_diff = temp - ideal_temp
    humidity_factor = 1 - 0.003 * (humidity - 60)
    combined_spoilage_factor = (q10 ** (temp_diff / 10)) * (1 + 0.004 * (humidity - 60))

    # Prediction
    input_features = np.array([[temp, humidity, temp_diff,
                                humidity_factor, combined_spoilage_factor]])

    prediction = model.predict(input_features)[0]

    # Waste classification
    waste_risk = "High" if prediction < 3 else "Low"

    # Output
    st.subheader("Results")
    st.write(f"Predicted Shelf Life: {prediction:.2f} days")
    st.write(f"Waste Risk: {waste_risk}")