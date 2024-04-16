import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

def load_data(filepath):
    data = pd.read_csv(filepath)
    # Ensure column names are treated as strings
    data.columns = data.columns.map(str)
    return data

def categorize_orientation(value):
    if value < -0.5:
        return 'Negative'
    elif value > 0.5:
        return 'Positive'
    else:
        return 'Neutral'

def preprocess_data(data):
    label_encoders = {}
    for column in ['Libellé de la commune', 'Département', 'Sexe', 'Nom Complet']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    data.fillna(data.mean(), inplace=True)
    return data, label_encoders

def add_time_features(data):
    future_data = pd.DataFrame()
    specified_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028]
    for year in specified_years:
        temp = data.copy()
        temp['year'] = year
        future_data = pd.concat([future_data, temp], axis=0)
    return future_data

def train_and_save_model(X_train, y_train, model_path):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model

def load_model(model_path):
    return joblib.load(model_path)

def make_predictions(model, X):
    return model.predict(X)

def run_predictions():
    data = load_data('./4-modeling/combined_all.csv')
    data, label_encoders = preprocess_data(data)
    data['Orientation'] = data['Orientation'].apply(categorize_orientation)
    label_encoder = LabelEncoder()
    data['Orientation'] = label_encoder.fit_transform(data['Orientation'])
    X = data.drop(['Orientation'], axis=1)
    y = data['Orientation']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = './model/random_forest_model.pkl'
    model = train_and_save_model(X_train, y_train, model_path)

    model = load_model(model_path)
    y_pred = make_predictions(model, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy}')

    future_data = add_time_features(data)
    X_future = future_data.drop(['Orientation'], axis=1)
    future_predictions = make_predictions(model, X_future)
    future_data['Predicted Orientation'] = label_encoder.inverse_transform(future_predictions)
    
    return future_data[['Libellé de la commune', 'year', 'Predicted Orientation']]

if __name__ == "__main__":
    predictions = run_predictions()
    # Save the predictions to a CSV file, ensuring headers are written as strings
    predictions.to_csv('predictions.csv', index=False)

