import streamlit as st
import pickle
import pandas as pd

def preprocess_input(input_data, scaler):
    # Define the expected feature names in the same order as they were during fitting
    expected_features = ['glucose_concentration', 'body_mass_index', 'age']
    
    # Create a DataFrame from the input data with expected feature names
    input_df = pd.DataFrame([input_data], columns=expected_features)
    
    # Scale the input data using the provided scaler
    scaled_input = scaler.transform(input_df)
    
    return scaled_input


def predict_diabetes(input_data, knn_model):
    prediction = knn_model.predict(input_data)
    return prediction

def show_predict_page():
    # Load the trained model and scaler
    with open('diabetes_model.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
        knn_model = loaded_data["model"]
        scaler = loaded_data["scaler"]

    # Streamlit app layout
    st.title("Diabetes Prediction App")

    # User input
    age = st.slider("Age", min_value=0, max_value=120, value=30)
    bmi = st.slider("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0)
    glucose = st.slider("Glucose Concentration", min_value=0.0, max_value=300.0, value=100.0)

    # Button for calculation
    if st.button("Calculate"):
        # Preprocess input data
        input_data = {'age': age, 'body_mass_index': bmi, 'glucose_concentration': glucose}
        scaled_input = preprocess_input(input_data, scaler)

        # Make prediction
        prediction = predict_diabetes(scaled_input, knn_model)

        # Display prediction result
        if prediction[0]:
            st.write("Prediction: Positive for Diabetes")
        else:
            st.write("Prediction: Negative for Diabetes")
