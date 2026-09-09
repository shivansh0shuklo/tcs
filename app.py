from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "Dataset" / "Data.csv"

FEATURE_COLUMNS = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History',
    'Gender',
    'Married',
    'Dependents_0',
    'Dependents_1',
    'Dependents_2',
    'Dependents_3+',
    'Education',
    'Self_Employed',
    'Property_Area_Rural',
    'Property_Area_Semiurban',
    'Property_Area_Urban',
]


def build_model() -> RandomForestClassifier:
    df = pd.read_csv(DATA_PATH).copy()
    df = df.drop(columns=['Loan_ID'], errors='ignore')

    for column in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0]).map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0]).map({'Yes': 1, 'No': 0})
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0]).astype(str)
    df['Education'] = df['Education'].fillna(df['Education'].mode()[0]).map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0]).map({'Yes': 1, 'No': 0})
    df['Property_Area'] = df['Property_Area'].fillna(df['Property_Area'].mode()[0])
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    for column in ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
        df[column] = df[column].fillna(df[column].median())

    rows = []
    for _, row in df.iterrows():
        row_map = {
            'ApplicantIncome': float(row['ApplicantIncome']),
            'CoapplicantIncome': float(row['CoapplicantIncome']),
            'LoanAmount': float(row['LoanAmount']),
            'Loan_Amount_Term': float(row['Loan_Amount_Term']),
            'Credit_History': float(row['Credit_History']),
            'Gender': float(row['Gender']),
            'Married': float(row['Married']),
            'Dependents_0': 1.0 if str(row['Dependents']) == '0' else 0.0,
            'Dependents_1': 1.0 if str(row['Dependents']) == '1' else 0.0,
            'Dependents_2': 1.0 if str(row['Dependents']) == '2' else 0.0,
            'Dependents_3+': 1.0 if str(row['Dependents']) == '3+' else 0.0,
            'Education': float(row['Education']),
            'Self_Employed': float(row['Self_Employed']),
            'Property_Area_Rural': 1.0 if row['Property_Area'] == 'Rural' else 0.0,
            'Property_Area_Semiurban': 1.0 if row['Property_Area'] == 'Semiurban' else 0.0,
            'Property_Area_Urban': 1.0 if row['Property_Area'] == 'Urban' else 0.0,
        }
        rows.append(row_map)

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = df['Loan_Status'].astype(int)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    with MODEL_PATH.open('wb') as model_file:
        pickle.dump(model, model_file)

    return model


def load_model() -> RandomForestClassifier:
    if MODEL_PATH.exists():
        try:
            with MODEL_PATH.open('rb') as model_file:
                model = pickle.load(model_file)
            if hasattr(model, 'feature_names_in_'):
                actual_columns = list(model.feature_names_in_)
                if actual_columns != FEATURE_COLUMNS:
                    return build_model()
            else:
                return build_model()
            return model
        except Exception:
            pass
    return build_model()


app = Flask(__name__)
model = load_model()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = {
        'ApplicantIncome': float(request.form['ApplicantIncome']),
        'CoapplicantIncome': float(request.form['CoapplicantIncome']),
        'LoanAmount': float(request.form['LoanAmount']),
        'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
        'Credit_History': float(request.form['Credit_History']),
        'Gender': float(request.form['Gender']),
        'Married': float(request.form['Married']),
        'Dependents_0': float(request.form['Dependents_0']),
        'Dependents_1': float(request.form['Dependents_1']),
        'Dependents_2': float(request.form['Dependents_2']),
        'Dependents_3+': float(request.form['Dependents_3+']),
        'Education': float(request.form['Education']),
        'Self_Employed': float(request.form['Self_Employed']),
        'Property_Area_Rural': float(request.form['Property_Area_Rural']),
        'Property_Area_Semiurban': float(request.form['Property_Area_Semiurban']),
        'Property_Area_Urban': float(request.form['Property_Area_Urban']),
    }

    feature_values = [features[column] for column in FEATURE_COLUMNS]
    prediction = model.predict(pd.DataFrame([feature_values], columns=FEATURE_COLUMNS))[0]
    result = 'Approved' if int(prediction) == 1 else 'Rejected'

    return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
