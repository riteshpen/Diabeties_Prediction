import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

__data_columns = None
__model = None
__scaler = None

def check_for_diabetes(Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age):
    global __model
    global __data_columns
    global __scaler

    # Ensure the model and data columns are loaded
    if __model is None or __data_columns is None or __scaler is None:
        load_saved_artifacts()

    # Create a numpy array for the input features
    input_features = np.array([[Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age]])

    # Scale the input features using the same scaler fitted during training
    input_features_scaled = __scaler.transform(input_features)

    # Predict the outcome
    prediction = __model.predict(input_features_scaled)
    return prediction[0]
def get_demographic_info():
    return __data_columns[:1]
def load_saved_artifacts():
    print('Loading saved artifacts ...start')
    global __data_columns
    global __model
    global __scaler

    # Load data columns and model
    try:
        with open('modeL-json', 'r') as f:
            __data_columns = json.load(f)['data_columns']
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    try:
        with open('model-pickle', 'rb') as f:
            __model = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load the scaler used during training
    try:
        with open('scaler-pickle', 'rb') as f:
            __scaler = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print('Loading saved artifacts ...done')

if __name__ == '__main__':
    load_saved_artifacts()

    if __data_columns:
        # Test the prediction function
        prediction = check_for_diabetes(6, 148, 72, 33.6, 0.627, 50)
        print(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
    else:
        print("Failed to load the data columns.")
