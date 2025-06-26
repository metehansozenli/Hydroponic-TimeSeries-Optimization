import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
import base64

# Append the parent directory to path to import from the main project
import sys
sys.path.append('..')
from app import app, predictor

def save_model():
    """
    Save the LSTM model for the simulator from the original file.
    This should be run only once.
    """
    try:
        # Import the model from the parent directory
        from lstm_plant_growth_model_optimum_env import model
        model.save('plant_growth_lstm_model.h5')
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    # Check if model exists, if not try to save it
    model_path = '../plant_growth_lstm_model.h5'
    if not os.path.exists(model_path):
        print("Model file not found, attempting to save...")
        save_model()
    
    # Check if scalers.json exists
    scaler_path = '../scalers.json'
    if not os.path.exists(scaler_path):
        print("Scalers file not found, please run save_scalers.py first")
        exit(1)
    
    # Run the Flask app
    print("Starting Plant Growth Simulator...")
    app.run(debug=True, port=5001)
