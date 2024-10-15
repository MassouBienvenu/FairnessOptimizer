import os
from flask import Flask, redirect, request, jsonify, send_from_directory, url_for
from main import FairnessOptimizer
import pandas as pd
import json
from flask import render_template
app = Flask(__name__)
@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(os.getcwd(), filename, as_attachment=True)

@app.route('/results')
def results():
    return render_template('results.html')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize_fairness():
    print("Received optimization request")
    try:
        config = json.loads(request.form['config'])
        print(f"Received config: {config}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON config: {str(e)}")
        return jsonify({"error": "Invalid JSON format", "message": str(e)}), 400

    data_file = request.files['data']
    print(f"Received data file: {data_file.filename}")
    
    # Read the CSV file directly from the uploaded file object
    df = pd.read_csv(data_file)
    print(f"Data loaded. Shape: {df.shape}")
    
    # Check for required configuration parameters
    required_params = ['sensitive_attributes', 'coefficient']
    missing_params = [param for param in required_params if param not in config]
    if missing_params:
        error_msg = f"Missing required parameters in config: {', '.join(missing_params)}"
        print(error_msg)
        return jsonify({"error": error_msg}), 400
    
    print("Initializing FairnessOptimizer...")
    optimizer = FairnessOptimizer(config, df)
    print("Starting optimization process...")
    dataset_path, report_path, adjusted_data = optimizer.optimize_fairness()
    
    print("Optimization completed. Preparing response...")
    
    response = {
        'optimized_dataset_path': dataset_path,
        'report_path': report_path,
        'adjusted_data_shape': adjusted_data.shape
    }

    print(f"Sending response: {response}")
    return redirect(url_for('results'))
if __name__ == '__main__':
    app.run(debug=True)
