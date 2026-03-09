from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

app = Flask(__name__)
app.secret_key = 'mindbloom_secret_2024'

MODEL_PATH = 'model/knn_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
ENCODER_PATH = 'model/encoder.pkl'

def train_and_save_model():
    os.makedirs('model', exist_ok=True)
    df = pd.read_csv('Dataset.csv')
    df = df.dropna()

    le_gender = LabelEncoder()
    df['Gender_enc'] = le_gender.fit_transform(df['Gender'])

    features = [
        'Age', 'Gender_enc', 'Sleep_Hours', 'Work_Study_Hours',
        'Physical_Activity_Days_Per_Week', 'Social_Interaction_Score_0_10',
        'Stress_Level_1_10', 'Anxiety_Score_0_21', 'Depression_Score_0_27',
        'Screen_Time_Hours', 'Caffeine_Intake_0_5', 'Alcohol_Consumption_0_5',
        'Family_History_0_1', 'Therapy_History_0_1'
    ]

    X = df[features].values
    y = df['Mental_Health_Condition'].values

    le_cond = LabelEncoder()
    y_enc = le_cond.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', algorithm='ball_tree', n_jobs=-1)
    knn.fit(X_train, y_train)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(knn, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump({'label_encoder': le_cond, 'gender_encoder': le_gender}, f)

    print(f"Model trained. Accuracy: {knn.score(X_test, y_test):.2f}")

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        knn = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        encoders = pickle.load(f)
    return knn, scaler, encoders

# Train model on startup if not exists
if not os.path.exists(MODEL_PATH):
    train_and_save_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/welcome2')
def welcome2():
    return render_template('welcome2.html')

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/lifestyle')
def lifestyle():
    return render_template('lifestyle.html')

@app.route('/gad7')
def gad7():
    return render_template('gad7.html')

@app.route('/phq9')
def phq9():
    return render_template('phq9.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/end')
def end():
    return render_template('end.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        knn, scaler, encoders = load_model()
        le_cond = encoders['label_encoder']
        le_gender = encoders['gender_encoder']

        gender_val = data.get('gender', 'Other')
        try:
            gender_enc = le_gender.transform([gender_val])[0]
        except:
            gender_enc = 0

        features = np.array([[
            float(data.get('age', 25)),
            float(gender_enc),
            float(data.get('sleep_hours', 2)),
            float(data.get('work_study_hours', 2)),
            float(data.get('physical_activity', 0)),
            float(data.get('social_interaction', 5)),
            float(data.get('stress_level', 5)),
            float(data.get('anxiety_score', 0)),
            float(data.get('depression_score', 0)),
            float(data.get('screen_time', 2)),
            float(data.get('caffeine', 0)),
            float(data.get('alcohol', 0)),
            float(data.get('family_history', 0)),
            float(data.get('therapy_history', 0))
        ]])

        features_scaled = scaler.transform(features)
        prediction_enc = knn.predict(features_scaled)[0]
        proba = knn.predict_proba(features_scaled)[0]
        condition = le_cond.inverse_transform([prediction_enc])[0]

        classes = le_cond.classes_
        proba_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}

        return jsonify({
            'condition': condition,
            'probabilities': proba_dict,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
