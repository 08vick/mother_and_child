# app.py - SDG 3 Health Risk Predictor (Web App)
from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)

# ----------------------------
# TRAIN MODELS ON STARTUP (or load pre-trained)
# ----------------------------
def train_models():
    np.random.seed(42)
    
    # --- Pregnancy Model ---
    n = 800
    X_preg = np.column_stack([
        np.random.randint(15, 45, n),      # age
        np.random.randint(0, 6, n),        # parity
        np.random.binomial(1, 0.3, n),     # anemia
        np.random.exponential(4, n),       # distance
        np.random.randint(0, 9, n),        # ANC visits
        np.random.binomial(1, 0.65, n)     # rural
    ])
    risk_score = 0.25*(X_preg[:,0]>35) + 0.2*(X_preg[:,1]>=4) + 0.3*X_preg[:,2] + 0.25*(X_preg[:,4]<4)
    y_preg = (risk_score + np.random.normal(0, 0.2, n)) > 0.6
    
    scaler_p = StandardScaler()
    X_preg_scaled = scaler_p.fit_transform(X_preg)
    model_p = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model_p.fit(X_preg_scaled, y_preg)
    
    # --- Vaccination Model ---
    n = 1200
    X_vacc = np.column_stack([
        np.random.randint(1, 24, n),       # age_months
        np.random.binomial(1, 0.2, n),     # low birth weight
        np.random.randint(0, 16, n),       # mother education
        np.random.exponential(4, n),       # distance
        np.random.binomial(1, 0.7, n),     # has health card
        np.random.binomial(1, 0.6, n)      # rural
    ])
    vacc_score = 0.3*X_vacc[:,5] + 0.3*(1-X_vacc[:,4]) + 0.25*(X_vacc[:,2]<6) + 0.15*X_vacc[:,1]
    y_vacc = (vacc_score + np.random.normal(0, 0.2, n)) > 0.55
    
    scaler_v = StandardScaler()
    X_vacc_scaled = scaler_v.fit_transform(X_vacc)
    model_v = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model_v.fit(X_vacc_scaled, y_vacc)
    
    return model_p, scaler_p, model_v, scaler_v

# Train models when app starts
model_preg, scaler_preg, model_vacc, scaler_vacc = train_models()

# ----------------------------
# ROUTES
# ----------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    task = data['task']
    
    if task == 'pregnancy':
        # [age, parity, anemia, distance, anc_visits, rural]
        features = np.array([[
            int(data['age']),
            int(data['parity']),
            1 if data['anemia'] == 'yes' else 0,
            float(data['distance']),
            int(data['anc_visits']),
            1 if data['rural'] == 'yes' else 0
        ]])
        features_scaled = scaler_preg.transform(features)
        prob = model_preg.predict_proba(features_scaled)[0][1]
        risk = "High Risk" if prob > 0.5 else "Low Risk"
        return jsonify({
            'result': f"{risk} pregnancy",
            'probability': f"{prob:.1%}"
        })
    
    elif task == 'vaccination':
        # [age_months, low_bw, mother_edu, distance, has_card, rural]
        features = np.array([[
            int(data['age_months']),
            1 if data['low_bw'] == 'yes' else 0,
            int(data['mother_edu']),
            float(data['distance']),
            1 if data['has_card'] == 'yes' else 0,
            1 if data['rural'] == 'yes' else 0
        ]])
        features_scaled = scaler_vacc.transform(features)
        prob = model_vacc.predict_proba(features_scaled)[0][1]
        status = "Incomplete" if prob > 0.5 else "Complete"
        return jsonify({
            'result': f"Vaccination schedule likely {status}",
            'probability': f"{prob:.1%}"
        })

if __name__ == '__main__':
    app.run(debug=True)