from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import os
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Get port from environment variable or default to 5000
port = int(os.environ.get('PORT', 5000))

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Global variables for model and data
df = None
kmeans = None
multi_target_model = None
imputer = None
scaler = None
feature_cols = None
X_test = None
y_test = None

def initialize_model():
    """Initialize the machine learning model and load data"""
    global df, kmeans, multi_target_model, imputer, scaler, feature_cols, X_test, y_test
    
    # Check if dataset exists, if not create sample data
    if not os.path.exists('heart_disease_uci.csv'):
        print("Dataset not found. Creating sample dataset...")
        # Create sample dataset for demonstration
        np.random.seed(42)
        n_samples = 1000
        ages = np.random.randint(25, 80, n_samples)
        sexes = np.random.choice([0, 1], n_samples)
        blood_pressures = np.random.randint(90, 200, n_samples)
        cholesterols = np.random.randint(120, 350, n_samples)
        max_heart_rates = np.random.randint(60, 200, n_samples)
        fasting_blood_sugars = np.random.choice([True, False], n_samples, p=[0.2, 0.8])
        
        # Create heart disease stages based on some logic
        heart_disease_stages = []
        for i in range(n_samples):
            risk_factors = 0
            if ages[i] > 50: risk_factors += 1
            if blood_pressures[i] > 140: risk_factors += 1
            if cholesterols[i] > 240: risk_factors += 1
            if max_heart_rates[i] < 100: risk_factors += 1
            if fasting_blood_sugars[i]: risk_factors += 1
            
            stage = min(risk_factors, 4)  # Cap at 4
            heart_disease_stages.append(stage)
        
        # Create DataFrame
        sample_df = pd.DataFrame({
            'age': ages,
            'sex': ['Male' if s == 1 else 'Female' for s in sexes],
            'trestbps': blood_pressures,
            'chol': cholesterols,
            'thalch': max_heart_rates,
            'fbs': fasting_blood_sugars,
            'num': heart_disease_stages
        })
        sample_df.to_csv('heart_disease_uci.csv', index=False)
        print("Sample dataset created successfully.")

    # Load data
    df = pd.read_csv('heart_disease_uci.csv')

    # Preprocessing
    df['Sex'] = df['sex'].map({'Male': 1, 'Female': 0})  # type: ignore
    df['High_Fasting_BloodSugar'] = df['fbs'].map({True: 1, False: 0})  # type: ignore
    df.rename(columns={'age': 'Age', 'trestbps': 'BloodPressure', 'chol': 'Cholesterol',
                       'thalch': 'MaxHeartRate', 'num': 'HeartDisease_Stage'}, inplace=True)

    feature_cols = ['Age', 'Sex', 'BloodPressure', 'Cholesterol', 'MaxHeartRate', 'High_Fasting_BloodSugar']
    imputer = SimpleImputer(strategy='mean')
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    # Targets
    df['Has_HeartDisease'] = df['HeartDisease_Stage'].apply(lambda x: 1 if x > 0 else 0)
    df['Has_Hypertension'] = df['BloodPressure'].apply(lambda x: 1 if x > 130 else 0)
    df['Has_Diabetes_Risk'] = df['High_Fasting_BloodSugar'].astype(int)

    # Training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    # Unsupervised (Clusters)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # type: ignore
    df['Patient_Profile_Cluster'] = kmeans.fit_predict(X_scaled)

    # Supervised (Prediction)
    X = df[feature_cols + ['Patient_Profile_Cluster']]
    y = df[['Has_HeartDisease', 'Has_Hypertension', 'Has_Diabetes_Risk']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    multi_target_model = MultiOutputClassifier(rf, n_jobs=-1)
    multi_target_model.fit(X_train, y_train)

def save_prediction(patient_data, predictions, cluster_id):
    """Save patient prediction to history file"""
    history_file = 'patient_history.json'
    
    # Create patient record
    record = {
        'timestamp': datetime.now().isoformat(),
        'patient_data': patient_data,
        'cluster_id': int(cluster_id),
        'predictions': {
            'heart_disease': bool(predictions[0]),
            'hypertension': bool(predictions[1]),
            'diabetes_risk': bool(predictions[2])
        }
    }
    
    # Load existing history or create new
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                content = f.read()
                if content:
                    history = json.loads(content)
        except (json.JSONDecodeError, IOError):
            history = []
    
    # Append new record
    history.append(record)
    
    # Save updated history
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    except IOError as e:
        print(f"Could not save prediction: {e}")

def create_dashboard(p_age, p_sex, p_bp, p_chol, p_hr, p_fbs, cluster_id, preds):
    """Create dashboard with visualizations"""
    labels = ['Heart Disease', 'Hypertension', 'Diabetes Risk']
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    plt.suptitle(f"Medical Insight Dashboard | Patient Age: {int(p_age)} | Sex: {'M' if p_sex==1 else 'F'}", 
                 fontsize=16, fontweight='bold')

    # [ROW 1, COL 1] PATIENT CONTEXT (Where do they fit?)
    sns.scatterplot(data=df, x='Age', y='MaxHeartRate', hue='Patient_Profile_Cluster',
                    palette='viridis', alpha=0.4, ax=axs[0, 0])  # type: ignore
    axs[0, 0].scatter(p_age, p_hr, color='red', s=300, marker='*', label='THIS PATIENT', zorder=10)  # type: ignore
    axs[0, 0].set_title("1. Patient Profile Context", fontweight='bold')
    axs[0, 0].legend(loc='upper right')

    # [ROW 1, COL 2] RELIABILITY CHECK (Confusion Matrix for Heart Disease)
    try:
        if multi_target_model is not None and hasattr(multi_target_model, 'estimators_') and len(multi_target_model.estimators_) > 0:
            y_pred_heart = multi_target_model.estimators_[0].predict(X_test)  # type: ignore
            cm = confusion_matrix(y_test['Has_HeartDisease'], y_pred_heart)  # type: ignore
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0, 1], cbar=False, annot_kws={"size": 14})  # type: ignore
            axs[0, 1].set_title("2. AI Reliability (Heart Disease)", fontweight='bold')
            axs[0, 1].set_xlabel("Predicted Diagnosis")
            axs[0, 1].set_ylabel("True Diagnosis")
            axs[0, 1].set_xticklabels(['Healthy', 'Sick'])
            axs[0, 1].set_yticklabels(['Healthy', 'Sick'])
    except (IndexError, AttributeError, TypeError):
        axs[0, 1].text(0.5, 0.5, "Data not available", ha='center', va='center')
        axs[0, 1].set_title("2. AI Reliability (Heart Disease)", fontweight='bold')

    # [ROW 1, COL 3] CONFIDENCE (Accuracy Bars)
    if multi_target_model is not None and hasattr(multi_target_model, 'predict'):
        try:
            y_pred = multi_target_model.predict(X_test)  # type: ignore
            accuracies = [accuracy_score(y_test[col], y_pred[:, i]) for i, col in enumerate(y_test.columns)]  # type: ignore
            bars = axs[0, 2].bar(labels, accuracies, color=['#3498db', '#e74c3c', '#f1c40f'])  # type: ignore
            axs[0, 2].set_title("3. System Confidence", fontweight='bold')
            axs[0, 2].set_ylim(0, 1.15)
            for i, (bar, v) in enumerate(zip(bars, accuracies)):
                axs[0, 2].text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.1%}", 
                              ha='center', fontweight='bold')
        except Exception:
            axs[0, 2].text(0.5, 0.5, "Data not available", ha='center', va='center')
            axs[0, 2].set_title("3. System Confidence", fontweight='bold')

    # [ROW 2, COL 1] EXPLAINABILITY (Feature Importance)
    try:
        if multi_target_model is not None and hasattr(multi_target_model, 'estimators_'):
            importances = np.mean([est.feature_importances_ for est in multi_target_model.estimators_], axis=0)  # type: ignore
            indices = np.argsort(importances)
            axs[1, 0].barh(range(len(indices)), importances[indices], color='#2ecc71', align='center')  # type: ignore
            axs[1, 0].set_yticks(range(len(indices)))
            axs[1, 0].set_yticklabels([X.columns[i] for i in indices])  # type: ignore
            axs[1, 0].set_title("4. Key Risk Factors", fontweight='bold')
            axs[1, 0].set_xlabel("Impact Score")
    except (AttributeError, IndexError, TypeError):
        axs[1, 0].text(0.5, 0.5, "Data not available", ha='center', va='center')
        axs[1, 0].set_title("4. Key Risk Factors", fontweight='bold')

    # [ROW 2, COL 2] RISK DISTRIBUTION
    risk_counts = df[['Has_HeartDisease', 'Has_Hypertension', 'Has_Diabetes_Risk']].sum()
    axs[1, 1].pie(risk_counts.values, labels=labels, autopct='%1.1f%%', 
                 colors=['#3498db', '#e74c3c', '#f1c40f'])  # type: ignore
    axs[1, 1].set_title("5. Population Risk Distribution", fontweight='bold')

    # [ROW 2, COL 3] PATIENT RISK PROFILE
    risk_levels = ['Low', 'Medium', 'High']
    patient_risks = [preds[0], preds[1], preds[2]]
    colors = ['#27ae60' if r == 0 else '#e74c3c' for r in patient_risks]
    axs[1, 2].barh(risk_levels, [1, 1, 1], color='lightgray', height=0.3)  # type: ignore
    axs[1, 2].barh(risk_levels, patient_risks, color=colors, height=0.3)  # type: ignore
    axs[1, 2].set_xlim(0, 1)
    axs[1, 2].set_title("6. Individual Risk Assessment", fontweight='bold')
    axs[1, 2].set_xlabel("Risk Level")
    for i, (risk, color) in enumerate(zip(patient_risks, colors)):
        axs[1, 2].text(0.5, i, 'DETECTED' if risk == 1 else 'NOT DETECTED', 
                      ha='center', va='center', fontweight='bold', 
                      color='white' if risk == 1 else 'black')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    
    # Convert plot to base64 for embedding in HTML
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        p_age = float(request.form['age'])
        p_sex = 1 if request.form['sex'] == 'Male' else 0
        p_bp = float(request.form['blood_pressure'])
        p_chol = float(request.form['cholesterol'])
        p_hr = float(request.form['max_heart_rate'])
        p_fbs = 1 if request.form['fasting_blood_sugar'] == 'Yes' else 0
        
        # Process prediction
        input_data = pd.DataFrame([[p_age, p_sex, p_bp, p_chol, p_hr, p_fbs]], columns=feature_cols)  # type: ignore
        filled = pd.DataFrame(imputer.transform(input_data), columns=feature_cols)  # type: ignore
        scaled = scaler.transform(filled)
        
        cluster_id = kmeans.predict(scaled)[0]  # type: ignore
        filled['Patient_Profile_Cluster'] = cluster_id
        preds = multi_target_model.predict(filled)[0]  # type: ignore
        
        # Save prediction
        patient_data = {
            'age': p_age,
            'sex': 'Male' if p_sex == 1 else 'Female',
            'blood_pressure': p_bp,
            'cholesterol': p_chol,
            'max_heart_rate': p_hr,
            'fasting_blood_sugar_high': bool(p_fbs)
        }
        save_prediction(patient_data, preds, cluster_id)
        
        # Create dashboard
        plot_url = create_dashboard(p_age, p_sex, p_bp, p_chol, p_hr, p_fbs, cluster_id, preds)
        
        # Prepare results
        results = {
            'cluster_id': int(cluster_id),
            'predictions': {
                'heart_disease': bool(preds[0]),
                'hypertension': bool(preds[1]),
                'diabetes_risk': bool(preds[2])
            }
        }
        
        return render_template('results.html', 
                             results=results, 
                             plot_url=plot_url,
                             patient_data=patient_data)
                             
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    # Initialize model
    initialize_model()
    
    # Run the app
    app.run(debug=False, host='0.0.0.0', port=port)