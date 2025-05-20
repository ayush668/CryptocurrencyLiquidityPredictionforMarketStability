import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Crypto Price Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the saved model
@st.cache_resource
def load_model():
    model = joblib.load('best_model_pipeline.joblib')
    return model

model = load_model()

# Feature names from your pipeline
def get_feature_names():
    return ['price', 'size_volume_ratio', 'is_stablecoin', 'vol_ratio_24h_7d', 
            'log_price', 'sqrt_vol_ratio', 'price_vol_interaction', 
            'price_bin', 'price_to_vol_ratio', 'is_high_volatility']

# Identify categorical features based on names
def get_categorical_features():
    # Based on feature names that suggest categorical data
    return ['is_stablecoin', 'price_bin', 'is_high_volatility']

features = get_feature_names()
categorical_features = get_categorical_features()

# Title and description
st.title("Cryptocurrency Price Prediction App")
st.write("Enter the cryptocurrency metrics below to get predictions from the trained model.")

# Create form for user input
with st.form("prediction_form"):
    st.subheader("Input Features")
    
    # Create columns for a cleaner layout
    cols = st.columns(2)
    
    # Create input fields for each feature
    input_data = {}
    
    for i, feature in enumerate(features):
        col_idx = i % 2
        with cols[col_idx]:
            if feature == 'is_stablecoin':
                input_data[feature] = st.selectbox(f"{feature}", [0, 1], 
                                                  format_func=lambda x: "Yes" if x == 1 else "No")
            
            elif feature == 'is_high_volatility':
                input_data[feature] = st.selectbox(f"{feature}", [0, 1],
                                                  format_func=lambda x: "High" if x == 1 else "Low")
            
            elif feature == 'price_bin':
                input_data[feature] = st.selectbox(f"{feature}", [0, 1, 2, 3, 4], 
                                                  format_func=lambda x: f"Category {x}")
            
            elif feature == 'price':
                input_data[feature] = st.number_input(f"{feature} (USD)", min_value=0.0, value=1000.0)
            
            elif 'ratio' in feature:
                input_data[feature] = st.number_input(f"{feature}", min_value=0.0, value=1.0)
            
            elif feature == 'log_price':
                # Calculate this automatically based on price
                pass
            
            elif feature == 'sqrt_vol_ratio':
                # Calculate this automatically based on vol_ratio
                pass
            
            elif feature == 'price_vol_interaction':
                # Calculate this automatically
                pass
            
            else:
                input_data[feature] = st.number_input(f"{feature}", value=0.0)
    
    # Submit button
    submit_button = st.form_submit_button("Predict")

# Calculate derived features when form is submitted
if submit_button:
    # Handle derived features
    if 'price' in input_data and 'log_price' in features:
        input_data['log_price'] = np.log1p(input_data['price'])
    
    if 'vol_ratio_24h_7d' in input_data and 'sqrt_vol_ratio' in features:
        input_data['sqrt_vol_ratio'] = np.sqrt(input_data['vol_ratio_24h_7d'])
    
    if 'price' in input_data and 'vol_ratio_24h_7d' in input_data and 'price_vol_interaction' in features:
        input_data['price_vol_interaction'] = input_data['price'] * input_data['vol_ratio_24h_7d']
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Show prediction results
    st.subheader("Prediction Results")
    st.write(f"Predicted Class: {prediction[0]}")
    
    # If model supports probability prediction
    try:
        probabilities = model.predict_proba(input_df)
        st.write("Class Probabilities:")
        prob_df = pd.DataFrame(
            probabilities, 
            columns=[f"Class {i}" for i in range(len(probabilities[0]))]
        )
        st.dataframe(prob_df)
        
        # Add visualization for probabilities
        st.bar_chart(prob_df.T)
    except:
        st.write("Probability scores not available for this model.")

# Add additional information
st.sidebar.header("About")
st.sidebar.info(
    """
    This app predicts cryptocurrency behavior based on various metrics.
    
    The model is a Random Forest Classifier with balanced subsample class weights.
    
    Features:
    - Price metrics
    - Volume ratios
    - Volatility indicators
    - Stablecoin status
    """
)

# Optional: Add feature to upload test data CSV for batch prediction
st.sidebar.header("Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV for batch prediction", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    test_data = pd.read_csv(uploaded_file)
    
    # Show the data
    st.subheader("Uploaded Data")
    st.dataframe(test_data)
    
    # Make batch predictions on button click
    if st.sidebar.button("Run Batch Prediction"):
        # Check if columns match
        missing_cols = set(features) - set(test_data.columns)
        if missing_cols:
            st.error(f"Missing columns in uploaded data: {missing_cols}")
        else:
            # Process any derived features in batch data
            if 'price' in test_data.columns and 'log_price' in features and 'log_price' not in test_data.columns:
                test_data['log_price'] = np.log1p(test_data['price'])
            
            if 'vol_ratio_24h_7d' in test_data.columns and 'sqrt_vol_ratio' in features and 'sqrt_vol_ratio' not in test_data.columns:
                test_data['sqrt_vol_ratio'] = np.sqrt(test_data['vol_ratio_24h_7d'])
            
            if 'price' in test_data.columns and 'vol_ratio_24h_7d' in test_data.columns and 'price_vol_interaction' in features and 'price_vol_interaction' not in test_data.columns:
                test_data['price_vol_interaction'] = test_data['price'] * test_data['vol_ratio_24h_7d']
            
            # Make prediction
            batch_predictions = model.predict(test_data)
            
            # Create results dataframe
            results = test_data.copy()
            results["Prediction"] = batch_predictions
            
            # Display results
            st.subheader("Prediction Results")
            st.dataframe(results)
            
            # Option to download results
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="crypto_prediction_results.csv",
                mime="text/csv"
            )