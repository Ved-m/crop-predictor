from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

_model = None
_le = None
_model_name = None
_accuracy = None
_crops = None

def load_model():
    global _model, _le, _model_name, _accuracy, _crops
    
    if _model is None:
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'crop_model.pkl')
            package = joblib.load(model_path)
            
            _model = package['model']
            _le = package['label_encoder']
            _model_name = package['model_name']
            _accuracy = package['accuracy']
            _crops = package['crops']
            
            print(f"✅ Model loaded: {_model_name}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    return _model, _le, _model_name, _accuracy, _crops

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        model, le, model_name, accuracy, crops = load_model()
        data = request.get_json()

        nitrogen = float(data['nitrogen'])
        phosphorus = float(data['phosphorus'])
        potassium = float(data['potassium'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        if not (0 <= ph <= 14):
            return jsonify({'success': False, 'error': 'pH must be 0-14'}), 400
        if not (0 <= humidity <= 100):
            return jsonify({'success': False, 'error': 'Humidity must be 0-100'}), 400
        if rainfall < 0:
            return jsonify({'success': False, 'error': 'Rainfall cannot be negative'}), 400

        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        prediction = model.predict(features)
        predicted_crop = le.inverse_transform(prediction)[0]

        confidence = None
        top_predictions = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = float(np.max(proba) * 100)
            top_indices = np.argsort(proba)[-3:][::-1]
            top_predictions = [
                {'crop': le.inverse_transform([idx])[0], 'probability': float(proba[idx] * 100)}
                for idx in top_indices
            ]

        return jsonify({
            'success': True,
            'crop': predicted_crop,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'model_info': {'name': model_name, 'accuracy': accuracy}
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
