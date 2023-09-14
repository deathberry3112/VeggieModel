from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        commodity_name = data['commodity_name']
        minimum_price = float(data['minimum_price'])
        maximum_price = float(data['maximum_price'])
        average_price = float(data['average_price'])

        commodity_encoded = label_encoder.transform([commodity_name])[0]

        scaled_features = scaler.transform([[commodity_encoded, minimum_price, maximum_price, average_price]])

        prediction = model.predict(scaled_features)

        response = {'prediction': prediction[0]}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
