import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle
from sklearn.preprocessing import MinMaxScaler

# Set page configuration
st.set_page_config(
    page_title="Weather Forecasting App",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-top: 2rem;
    }
    .input-box {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #757575;
        font-size: 0.9rem;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #FF9800;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    """Load the trained model with caching for better performance"""
    try:
        model = tf.keras.models.load_model(
            'weather_forecasting.h5',
            custom_objects={'mse': tf.keras.metrics.MeanSquaredError}
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_scaler():
    """Load the scaler with caching"""
    try:
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {str(e)}")
        return None

def prepare_input_sequence(temperature, humidity, wind_speed, pressure, scaler, sequence_length=24):
    """
    Prepare input sequence for prediction with proper scaling
    """
    # Create input data with the same structure as training (4 features)
    input_data = np.array([[temperature, humidity, wind_speed, pressure]])
    input_sequence = np.tile(input_data, (sequence_length, 1))
    
    # The scaler was trained on 5 features (including target), but we only have 4 features for prediction
    # We need to handle this mismatch
    
    # Create a dummy array with 5 features (like the training data)
    dummy_array = np.zeros((sequence_length, 5))
    
    # Fill the first 4 features with our input data
    dummy_array[:, :4] = input_sequence
    
    # Scale the data using the original scaler
    scaled_data = scaler.transform(dummy_array)
    
    # Extract only the first 4 features (our input features)
    scaled_input_sequence = scaled_data[:, :4]
    
    return scaled_input_sequence.reshape(1, sequence_length, 4)

# Load resources
model = load_trained_model()
scaler = load_scaler()

# App header
st.markdown('<h1 class="main-header">🌤️ Professional Weather Forecasting</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p>Input current weather parameters to predict the apparent temperature for the next hour.</p>
    <p>This model uses a CNN-LSTM architecture trained on historical weather data.</p>
</div>
""", unsafe_allow_html=True)

# Display warning if scaler is not loaded properly
if scaler is None:
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.warning("Scaler not loaded properly. Predictions may not be accurate.")
    st.markdown('</div>', unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📊 Input Parameters")
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    
    # Input parameters with better organization
    temperature = st.slider(
        "Temperature (°C)", 
        -20.0, 50.0, 24.7, 0.1,
        help="Current air temperature in Celsius"
    )
    
    humidity = st.slider(
        "Relative Humidity (%)", 
        0.0, 100.0, 50.0, 0.1,
        help="Percentage of moisture in the air"
    )
    
    wind_speed = st.slider(
        "Wind Speed (km/h)", 
        0.0, 100.0, 10.0, 0.1,
        help="Wind speed in kilometers per hour"
    )
    
    pressure = st.slider(
        "Atmospheric Pressure (hPa)", 
        950.0, 1050.0, 1013.0, 0.1,
        help="Atmospheric pressure in hectopascals (millibars)"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button
    predict_btn = st.button(
        "Predict Apparent Temperature", 
        type="primary", 
        use_container_width=True,
        help="Click to generate prediction based on input parameters"
    )

with col2:
    st.markdown("### 📋 Input Summary")
    
    # Display input in a structured way
    input_data = pd.DataFrame({
        'Parameter': ['Temperature', 'Humidity', 'Wind Speed', 'Pressure'],
        'Value': [f"{temperature} °C", f"{humidity}%", f"{wind_speed} km/h", f"{pressure} hPa"],
        'Units': ['°C', '%', 'km/h', 'hPa']
    })
    
    st.dataframe(
        input_data, 
        hide_index=True,
        use_container_width=True
    )
    
    # Prediction results section
    if predict_btn and model is not None and scaler is not None:
        try:
            # Prepare input sequence with proper scaling
            input_sequence = prepare_input_sequence(
                temperature, humidity/100, wind_speed, pressure, scaler
            )
            
            # Make prediction
            with st.spinner("Generating prediction..."):
                prediction = model.predict(input_sequence)
                predicted_temp = prediction[0][0]
                
                # Display prediction in a nicely styled box
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown("### 📈 Prediction Result")
                st.metric(
                    label="**Predicted Apparent Temperature**", 
                    value=f"{predicted_temp:.2f} °C",
                    delta=f"{(predicted_temp - temperature):.2f} °C difference from air temperature"
                )
                
                # Add some context about apparent temperature
                st.info("""
                **Apparent Temperature** is what the temperature feels like to the human body, 
                considering factors like humidity and wind speed. This can differ from the 
                actual air temperature.
                """)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("This error typically occurs when the scaler expects a different number of features than provided. The fix has been implemented in this version.")
    
    elif predict_btn:
        st.warning("Please ensure both model and scaler are properly loaded.")

# Add informational section
st.markdown("---")
st.markdown("### ℹ️ About This App")
st.markdown("""
This weather forecasting application uses a deep learning model (CNN-LSTM) to predict 
apparent temperature based on current weather conditions. The model was trained on 
historical weather data and considers:

- **Air Temperature** - Measured in Celsius
- **Relative Humidity** - Percentage of moisture in the air (converted to decimal for model input)
- **Wind Speed** - Measured in kilometers per hour
- **Atmospheric Pressure** - Measured in hectopascals

The apparent temperature represents what the temperature feels like to humans, 
which can differ from the actual air temperature due to factors like humidity and wind.

**Note**: Humidity is divided by 100 before being sent to the model to convert from percentage to decimal.
""")

# Footer
st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
Developed with ❤️ using Streamlit & TensorFlow | 
© 2023 Weather Forecasting App
""")
st.markdown('</div>', unsafe_allow_html=True)