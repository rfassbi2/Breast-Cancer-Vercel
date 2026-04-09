import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load trained model using absolute path (required for Vercel)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'trained_model.pkl')

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Breast Cancer Prediction App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 60px auto;
            padding: 0 20px;
            background-color: #f5f5f5;
        }
        h1 { color: #2c3e50; }
        label { display: block; margin-top: 16px; font-weight: bold; color: #333; }
        input[type=number] {
            width: 100%;
            padding: 8px;
            margin-top: 4px;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            margin-top: 24px;
            padding: 12px 24px;
            background-color: #2980b9;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
        }
        button:hover { background-color: #2471a3; }
        #result {
            margin-top: 24px;
            padding: 16px;
            border-radius: 6px;
            display: none;
        }
        .benign  { background-color: #d5f5e3; border: 1px solid #27ae60; color: #1e8449; }
        .malignant { background-color: #fadbd8; border: 1px solid #e74c3c; color: #c0392b; }
        .confidence { margin-top: 12px; font-size: 14px; color: #555; }
    </style>
</head>
<body>
    <h1>🔬 Breast Cancer Prediction App</h1>
    <p>Enter tumor measurements below:</p>

    <label>Mean Radius (6.0 – 30.0)
        <input type="number" id="mean_radius" value="15.0" min="6.0" max="30.0" step="0.1">
    </label>
    <label>Mean Texture (10.0 – 40.0)
        <input type="number" id="mean_texture" value="20.0" min="10.0" max="40.0" step="0.1">
    </label>
    <label>Mean Perimeter (40.0 – 200.0)
        <input type="number" id="mean_perimeter" value="100.0" min="40.0" max="200.0" step="0.1">
    </label>
    <label>Mean Area (150.0 – 2500.0)
        <input type="number" id="mean_area" value="800.0" min="150.0" max="2500.0" step="1">
    </label>
    <label>Mean Smoothness (0.05 – 0.20)
        <input type="number" id="mean_smoothness" value="0.1" min="0.05" max="0.20" step="0.001">
    </label>

    <button onclick="predict()">Predict</button>

    <div id="result">
        <strong id="verdict"></strong>
        <div class="confidence" id="confidence"></div>
    </div>

    <script>
        async function predict() {
            const data = {
                mean_radius:     parseFloat(document.getElementById('mean_radius').value),
                mean_texture:    parseFloat(document.getElementById('mean_texture').value),
                mean_perimeter:  parseFloat(document.getElementById('mean_perimeter').value),
                mean_area:       parseFloat(document.getElementById('mean_area').value),
                mean_smoothness: parseFloat(document.getElementById('mean_smoothness').value),
            };

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            const div = document.getElementById('result');
            const verdict = document.getElementById('verdict');
            const confidence = document.getElementById('confidence');

            div.style.display = 'block';
            div.className = result.prediction === 'Benign' ? 'benign' : 'malignant';
            verdict.textContent = result.prediction === 'Benign'
                ? '✅ Prediction: Benign (Not Cancer)'
                : '❌ Prediction: Malignant (Cancer)';
            confidence.innerHTML =
                `<strong>Confidence:</strong><br>
                 Benign: ${(result.prob_benign * 100).toFixed(1)}%<br>
                 Malignant: ${(result.prob_malignant * 100).toFixed(1)}%`;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features_df = pd.DataFrame([[
        data['mean_radius'],
        data['mean_texture'],
        data['mean_perimeter'],
        data['mean_area'],
        data['mean_smoothness']
    ]], columns=[
        'mean radius',
        'mean texture',
        'mean perimeter',
        'mean area',
        'mean smoothness'
    ])

    prediction = model.predict(features_df)
    probs = model.predict_proba(features_df)

    return jsonify({
        'prediction': 'Benign' if prediction[0] == 1 else 'Malignant',
        'prob_benign': round(float(probs[0][1]), 4),
        'prob_malignant': round(float(probs[0][0]), 4)
    })


if __name__ == '__main__':
    app.run(debug=True)
