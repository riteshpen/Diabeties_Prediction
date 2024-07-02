from flask import Flask, request, jsonify
from flask_cors import CORS
import util

app = Flask(import_name=__name__)
CORS(app)

@app.route('/get_num_pregnancies', methods=['GET'])
def get_num_pregancies():
    response = jsonify({
        'Pregnancies': util.get_demographic_info()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
@app.route('/check_for_diabetes', methods=['POST'])
def check_for_diabetes():
    try:
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])

        prediction = util.check_for_diabetes(Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age)

        response = jsonify({
            'diabetes_prediction': prediction
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        response = jsonify({'error': str(e)})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500

if __name__ == '__main__':
    print('Starting Python Flask for Diabeties Prediction')
    util.load_saved_artifacts()
    app.run(debug=True)