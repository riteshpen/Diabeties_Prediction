function onClickedPredictDiabetes() {
    console.log("Predict Diabetes button clicked");

    var pregnancies = parseInt(document.getElementById("Pregnancies").value);
    var glucose = parseInt(document.getElementById("Glucose").value);
    var bloodPressure = parseInt(document.getElementById("BloodPressure").value);
    var bmi = parseFloat(document.getElementById("BMI").value);
    var diabetesPedigreeFunction = parseFloat(document.getElementById("DiabetesPedigreeFunction").value);
    var age = parseInt(document.getElementById("Age").value);

    // Update the URL as needed
    var url = "http://127.0.0.1:5000/check_for_diabetes"; // Use this if you are running the server locally

    $.post(url, {
        Pregnancies: pregnancies,
        Glucose: glucose,
        BloodPressure: bloodPressure,
        BMI: bmi,
        DiabetesPedigreeFunction: diabetesPedigreeFunction,
        Age: age
    })
    .done(function(data) {
        var prediction = data.diabetes_prediction == 1 ? "Diabetic" : "Non-Diabetic";
        document.getElementById("predictionResult").innerHTML = "Prediction: " + prediction;
        console.log(data);
    })
    .fail(function(jqXHR, textStatus, errorThrown) {
        console.log("Error:", textStatus, errorThrown); // Log the error for debugging
        document.getElementById("predictionResult").innerHTML = "Prediction: Error predicting diabetes.";
    });
}

// No need for onPageLoad function for this context
