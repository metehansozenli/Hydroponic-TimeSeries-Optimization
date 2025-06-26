# Interactive Plant Growth Simulator
# Main entry point for running the simulator web app

import os
import sys
import json
import numpy as np
import subprocess
import time

def setup_simulator():
    """Set up the simulator environment"""
    print("Setting up Interactive Plant Growth Simulator...")
    
    # Check if scalers.json exists
    scaler_path = '../scalers.json'
    if not os.path.exists(scaler_path):
        print("Creating scalers file...")
        try:
            subprocess.run(['python', 'save_scalers.py'], check=True)
        except subprocess.CalledProcessError:
            print("Failed to create scalers. Make sure the original data file exists.")
            return False
    
    # Check if model file exists
    model_path = '../plant_growth_lstm_model.h5'
    if not os.path.exists(model_path):
        print("Model file not found. Attempting to extract model...")
        try:
            subprocess.run(['python', 'extract_model.py'], check=True)
        except subprocess.CalledProcessError:
            print("Failed to extract model. You may need to manually save it.")
            print("Open lstm_plant_growth_model_optimum_env.py and add:")
            print("model.save('plant_growth_lstm_model.h5')")
            return False
    
    # Check if plant image exists
    plant_img_path = 'static/images/plant.png'
    if not os.path.exists(plant_img_path):
        print("Creating plant image...")
        try:
            subprocess.run(['python', 'create_plant_image.py'], check=True)
        except subprocess.CalledProcessError:
            print("Failed to create plant image.")
    
    return True

def run_webapp():
    """Run the Flask web application"""
    print("Starting Flask web server...")
    try:
        from app import app
        app.run(debug=True, port=5001)
    except ImportError:
        print("Failed to import Flask application. Make sure app.py exists.")
        return False
    except Exception as e:
        print(f"Error starting web server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Interactive Plant Growth Simulator - Başlatılıyor")
    print("=" * 60)
    
    if setup_simulator():
        print("\nSimulator setup completed successfully!")
        print("Starting web application...\n")
        run_webapp()
    else:
        print("\nSetup failed. Please check the error messages above.")
        sys.exit(1)
