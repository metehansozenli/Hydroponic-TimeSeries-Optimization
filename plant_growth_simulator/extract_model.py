import sys
import os
import json
import numpy as np
from tensorflow.keras.models import load_model, save_model

# Add parent directory to path
sys.path.append('..')

def extract_and_save_model():
    """
    Extract and save the trained LSTM model from lstm_plant_growth_model_optimum_env.py
    """
    try:
        # Try to import the model directly
        print("Attempting to import model from parent directory...")
        
        # First option: try to import the model from the module
        try:
            from lstm_plant_growth_model_optimum_env import model
            model.save('../plant_growth_lstm_model.h5')
            print("Successfully imported and saved model.")
            return True
        except ImportError as e:
            print(f"Could not import model directly: {e}")
        
        # Second option: try to load and run the script to get access to the model
        print("Trying alternative method...")
        import tensorflow as tf
        
        # Define a simple function to capture the model
        stored_model = None
        
        def mock_save_model(model, path, **kwargs):
            global stored_model
            stored_model = model
            print(f"Captured model: {model}")
            # Don't actually save here
            return None
        
        # Replace tf.keras.models.save_model temporarily
        original_save = tf.keras.models.save_model
        tf.keras.models.save_model = mock_save_model
        
        # Execute the script to build the model
        exec(open('../lstm_plant_growth_model_optimum_env.py').read())
        
        # Restore original function
        tf.keras.models.save_model = original_save
        
        # Now save the captured model
        if stored_model is not None:
            stored_model.save('../plant_growth_lstm_model.h5')
            print("Successfully executed script and saved model.")
            return True
        else:
            print("Failed to capture model from script execution.")
            return False
            
    except Exception as e:
        print(f"Error extracting and saving model: {e}")
        return False

if __name__ == "__main__":
    if extract_and_save_model():
        print("Model extraction completed successfully.")
    else:
        print("Model extraction failed. You may need to manually save the model.")
        print("Please open lstm_plant_growth_model_optimum_env.py and add:")
        print("model.save('plant_growth_lstm_model.h5')")
        print("at the end of the script, then run it.")
