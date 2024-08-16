import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle
import json

# Load the dataset
df = pd.read_csv('diabetes (1).csv')

# Title of the Streamlit App
st.title("Diabetes Prediction and Data Analysis")

# Sidebar for user inputs
st.sidebar.header("Input Features")
Pregnancies = st.sidebar.slider("Pregnancies", int(df.Pregnancies.min()), int(df.Pregnancies.max()), int(df.Pregnancies.mean()))
Glucose = st.sidebar.slider("Glucose", int(df.Glucose.min()), int(df.Glucose.max()), int(df.Glucose.mean()))
BloodPressure = st.sidebar.slider("Blood Pressure", int(df.BloodPressure.min()), int(df.BloodPressure.max()), int(df.BloodPressure.mean()))
BMI = st.sidebar.slider("BMI", float(df.BMI.min()), float(df.BMI.max()), float(df.BMI.mean()))
DiabetesPedigreeFunction = st.sidebar.slider("Diabetes Pedigree Function", float(df.DiabetesPedigreeFunction.min()), float(df.DiabetesPedigreeFunction.max()), float(df.DiabetesPedigreeFunction.mean()))
Age = st.sidebar.slider("Age", int(df.Age.min()), int(df.Age.max()), int(df.Age.mean()))

# Plotting
st.subheader("Data Distribution")
columns_to_plot = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

# Plot count plots for each column
for i, ax in enumerate(axes.flatten()):
    if i < len(columns_to_plot):
        sns.countplot(x=df[columns_to_plot[i]], ax=ax)
        ax.set_title(f'Value Counts of {columns_to_plot[i]}')
        ax.set_xlabel(columns_to_plot[i])
        ax.set_ylabel('Count')
plt.tight_layout()
st.pyplot(fig)

# Pie chart for diabetes rate by number of pregnancies
st.subheader("Diabetes Cases vs. Number of Pregnancies")
pregnant_diabetes = df.groupby(['Pregnancies'])['Outcome'].mean()
plt.figure(figsize=(8, 6))
plt.pie(pregnant_diabetes, labels=pregnant_diabetes.index, autopct='%1.1f%%', shadow=True)
plt.title('Diabetes Cases versus Number of Pregnancies')
st.pyplot(plt)

# Histograms for each feature
st.subheader("Feature Histograms")
fig = plt.figure(figsize=(20, 15))
df.hist(bins=30, edgecolor='black', ax=fig.gca())
plt.tight_layout()
st.pyplot(fig)

# Check for NA's
st.write("Missing values in dataset:")
st.write(df.isna().sum())

# Check for Outliers and remove them
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_outlier = Q1 - (1.5 * IQR)
    upper_outlier = Q3 + (1.5 * IQR)
    return df[(df[column] > lower_outlier) & (df[column] < upper_outlier)]

df = remove_outliers(df, 'Insulin')
df = remove_outliers(df, 'Pregnancies')
df = remove_outliers(df, 'Glucose')

st.write("Data description after removing outliers:")
st.write(df.describe())

# Predictions and Fitting Models
x = df.drop(['Outcome', 'SkinThickness', 'Insulin'], axis='columns')
y = df['Outcome']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

model_params = {
    'svm': {
        'model': SVC(gamma='auto'),
        'params': {
            'C': [1, 10, 20],
            'kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [1, 5, 10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'C': [1, 5, 10]
        }
    },
    'decision_tree': {
        'model': tree.DecisionTreeRegressor(),
        'params': {
            'criterion': ['mse', 'friedman_mse'],
            'splitter': ['best', 'random']
        }
    }
}

# Perform grid search for each model
scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(x_scaled, y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

# Create a DataFrame with the results
results_df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
st.write("Model Results:")
st.write(results_df)

# Train final model for prediction
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2)
model = SVC(C=1, kernel='rbf')
model.fit(x_train, y_train)

# Prediction function
def predict_diabetes(Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age):
    x = np.array([[Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age]])
    x_scaled = scaler.transform(x)
    return model.predict(x_scaled)[0]

# Show prediction result on the web app
if st.sidebar.button("Predict Diabetes"):
    prediction = predict_diabetes(Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age)
    if prediction == 1:
        st.sidebar.write("Prediction: The person is likely to have diabetes.")
    else:
        st.sidebar.write("Prediction: The person is unlikely to have diabetes.")

# Save the model
with open('model-pickle.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler-pickle.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the columns
columns = {
    'data_columns': [col.lower() for col in x.columns]
}
with open('model-json', 'w') as f:
    f.write(json.dumps(columns))
