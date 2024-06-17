import pandas as pd
from sklearn.naive_bayes import GaussianNB
import joblib

# Load your dataset
data = pd.read_csv("seattle-weather.csv")  # Replace with your file path

# Assuming 'precipitation', 'temp_max', 'temp_min', 'wind' are numerical features
X = data[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = data['weather']

# Initialize and train the Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'trained_model.joblib')
