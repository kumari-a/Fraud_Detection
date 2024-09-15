from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl') 

def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data], columns=['type', 'amount', 'oldbalanceOrg', 'newbalanceDest', 'isFlaggedFraud'])

    input_scaled = scaler.transform(input_df)

    return input_scaled

def predict(input_data):
    input_processed = preprocess_input(input_data)
    y_pred = model.predict(input_processed)


    y_pred_class = y_pred.argmax(axis=1)

    return y_pred_class[0]  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    # Collect form data
    input_data = {
        'type': int(request.form['type']),
        'amount': float(request.form['amount']),
        'oldbalanceOrg': float(request.form['oldbalanceOrg']),
        'newbalanceDest': float(request.form['newbalanceDest']),
        'isFlaggedFraud': int(request.form['isFlaggedFraud'])
    }


    prediction = predict(input_data)


    return render_template('index.html', prediction=int(prediction))

if __name__ == '__main__':
    app.run(debug=True)
