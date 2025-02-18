from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle pickle
model = joblib.load('logistic_regression_model.pkl')

# Route principale pour afficher l'interface utilisateur
@app.route('/', methods=['GET'])
def home():
    return 'Running !'


@app.route('/predict', methods=['POST'])
def predict():
    try : 
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e : 
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
