import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load trained model
# Load the trained model
model = tf.keras.models.load_model('weather_forecasting.h5')

# Load the encoders and scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the MinMaxScaler (use the same settings as training)
scaler = MinMaxScaler(feature_range=(0, 1))

# Streamlit app
st.title("Weather Forecasting: Apparent Temperature Prediction")
st.write("Input weather parameters to predict the apparent temperature.")

# Input parameters
st.sidebar.header("Input Weather Data")
temperature = st.sidebar.slider("Temperature (C)", -10.0, 40.0, 20.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0)
pressure = st.sidebar.slider("Pressure (millibars)", 950.0, 1050.0, 1013.0)

# Display input
st.write("### Input Data")
st.write(f"- **Temperature**: {temperature} °C")
st.write(f"- **Humidity**: {humidity} %")
st.write(f"- **Wind Speed**: {wind_speed} km/h")
st.write(f"- **Pressure**: {pressure} millibars")

# Create a sample input sequence (24 hours with identical values for simplicity)
sequence_length = 24
input_data = np.array([[temperature, humidity / 100, wind_speed, pressure]])
input_sequence = np.tile(input_data, (sequence_length, 1))

# Normalize the input data
input_sequence = scaler.fit_transform(input_sequence)
input_sequence = input_sequence.reshape(1, sequence_length, 4)

# Make a prediction
if st.button("Predict Apparent Temperature"):
    prediction = model.predict(input_sequence)
    predicted_temp = prediction[0][0]  # Scaled output
    st.write(f"### Predicted Apparent Temperature: {predicted_temp:.2f} °C")

# Footer
st.write("Developed with ❤️ using Streamlit")
