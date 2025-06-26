from flask import Flask, render_template, request, jsonify
from growth_predictor import GrowthPredictor
import os
import json

app = Flask(__name__, static_folder='static', template_folder='templates')

# Initialize the growth predictor
predictor = GrowthPredictor()

@app.route('/')
def index():
    """Render the main page of the application."""
    return render_template('index.html', 
                          optimal_temp=predictor.optimal_temp,
                          optimal_ph=predictor.optimal_ph,
                          optimal_ec=predictor.optimal_ec,
                          temp_min=predictor.bounds['h2o_temp_C'][0],
                          temp_max=predictor.bounds['h2o_temp_C'][1],
                          ph_min=predictor.bounds['pH'][0],
                          ph_max=predictor.bounds['pH'][1],
                          ec_min=predictor.bounds['EC'][0],
                          ec_max=predictor.bounds['EC'][1])

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request and return growth data."""
    # Get parameters from form
    h2o_temp = float(request.form.get('h2o_temp'))
    ph = float(request.form.get('ph'))
    ec = float(request.form.get('ec'))
    days = int(request.form.get('days', 30))
    plant_type = request.form.get('plant_type', 'basil')  # Default to basil
    
    # Generate prediction
    growth_data = predictor.predict_growth(h2o_temp, ph, ec, days, plant_type)
    
    # Add chart image
    growth_data['chart_img'] = predictor.generate_growth_chart(growth_data)
    
    # Add environment score
    growth_data['env_score'] = predictor.evaluate_conditions(h2o_temp, ph, ec)
    
    return jsonify(growth_data)

@app.route('/optimal-comparison', methods=['GET'])
def optimal_comparison():
    """Return the optimal parameters for comparison."""
    return jsonify({
        'temperature': predictor.optimal_temp,
        'ph': predictor.optimal_ph,
        'ec': predictor.optimal_ec
    })

if __name__ == '__main__':
    # Save scalers.json if it doesn't exist
    if not os.path.exists('../scalers.json'):
        # Create a simple version for testing
        scalers_dict = {
            'scaler_X': {
                'center': [22.5, 6.3, 2.5],  # Mean values for h2o_temp_C, pH, EC
                'scale': [2, 0.5, 0.5]       # Std dev estimates
            },
            'scaler_Y': {
                'center': [30],  # Mean height
                'scale': [10]    # Std dev estimate
            }
        }
        with open('../scalers.json', 'w') as f:
            json.dump(scalers_dict, f)
    
    app.run(debug=True, port=5001)
