from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("trained_model.joblib")  # Replace with the path to your trained model

@app.route('/')
def home():
    return render_template('newnew.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'precipitation': float(request.form['precipitation']),
        'temp_max': float(request.form['temp_max']),
        'temp_min': float(request.form['temp_min']),
        'wind': float(request.form['wind'])
    }

    input_df = pd.DataFrame(input_data, index=[0])

    predicted_label = model.predict(input_df)
    return render_template('newnew.html', prediction=predicted_label[0])

if __name__ == '__main__':
    app.run()
