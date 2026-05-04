import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
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

# Set page configuration
st.set_page_config(
    page_title="Medical Diagnosis Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.df = None
    st.session_state.kmeans = None
    st.session_state.multi_target_model = None
    st.session_state.imputer = None
    st.session_state.scaler = None
    st.session_state.feature_cols = None
    st.session_state.X_test = None
    st.session_state.y_test = None

def initialize_model():
    """Initialize the machine learning model and load data"""
    if st.session_state.initialized:
        return
    
    # Check if dataset exists, if not create sample data
    if not os.path.exists('heart_disease_uci.csv'):
        st.info("Dataset not found. Creating sample dataset...")
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
        st.success("Sample dataset created successfully!")

    # Load data
    st.session_state.df = pd.read_csv('heart_disease_uci.csv')

    # Preprocessing
    st.session_state.df['Sex'] = st.session_state.df['sex'].map({'Male': 1, 'Female': 0})
    st.session_state.df['High_Fasting_BloodSugar'] = st.session_state.df['fbs'].map({True: 1, False: 0})
    st.session_state.df.rename(columns={'age': 'Age', 'trestbps': 'BloodPressure', 'chol': 'Cholesterol',
                       'thalch': 'MaxHeartRate', 'num': 'HeartDisease_Stage'}, inplace=True)

    st.session_state.feature_cols = ['Age', 'Sex', 'BloodPressure', 'Cholesterol', 'MaxHeartRate', 'High_Fasting_BloodSugar']
    st.session_state.imputer = SimpleImputer(strategy='mean')
    st.session_state.df[st.session_state.feature_cols] = st.session_state.imputer.fit_transform(st.session_state.df[st.session_state.feature_cols])

    # Targets
    st.session_state.df['Has_HeartDisease'] = st.session_state.df['HeartDisease_Stage'].apply(lambda x: 1 if x > 0 else 0)
    st.session_state.df['Has_Hypertension'] = st.session_state.df['BloodPressure'].apply(lambda x: 1 if x > 130 else 0)
    st.session_state.df['Has_Diabetes_Risk'] = st.session_state.df['High_Fasting_BloodSugar'].astype(int)

    # Training
    st.session_state.scaler = StandardScaler()
    X_scaled = st.session_state.scaler.fit_transform(st.session_state.df[st.session_state.feature_cols])

    # Unsupervised (Clusters)
    st.session_state.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    st.session_state.df['Patient_Profile_Cluster'] = st.session_state.kmeans.fit_predict(X_scaled)

    # Supervised (Prediction)
    X = st.session_state.df[st.session_state.feature_cols + ['Patient_Profile_Cluster']]
    y = st.session_state.df[['Has_HeartDisease', 'Has_Hypertension', 'Has_Diabetes_Risk']]
    st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    st.session_state.multi_target_model = MultiOutputClassifier(rf, n_jobs=-1)
    st.session_state.multi_target_model.fit(st.session_state.X_train, st.session_state.y_train)
    
    st.session_state.initialized = True

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
        st.error(f"Could not save prediction: {e}")

def create_dashboard(p_age, p_sex, p_bp, p_chol, p_hr, p_fbs, cluster_id, preds):
    """Create dashboard with visualizations using Plotly"""
    labels = ['Heart Disease', 'Hypertension', 'Diabetes Risk']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "1. Patient Profile Context",
            "2. AI Reliability (Heart Disease)",
            "3. System Confidence",
            "4. Key Risk Factors",
            "5. Population Risk Distribution",
            "6. Individual Risk Assessment"
        ),
        specs=[[{"type": "scatter"}, {"type": "heatmap"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "pie"}, {"type": "bar"}]],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    # [ROW 1, COL 1] PATIENT CONTEXT (Where do they fit?)
    for cluster in st.session_state.df['Patient_Profile_Cluster'].unique():
        cluster_data = st.session_state.df[st.session_state.df['Patient_Profile_Cluster'] == cluster]
        fig.add_trace(
            go.Scatter(
                x=cluster_data['Age'],
                y=cluster_data['MaxHeartRate'],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(size=8, opacity=0.6)
            ),
            row=1, col=1
        )
    
    # Add patient point
    fig.add_trace(
        go.Scatter(
            x=[p_age],
            y=[p_hr],
            mode='markers',
            name='THIS PATIENT',
            marker=dict(size=15, color='red', symbol='star')
        ),
        row=1, col=1
    )
    
    fig.update_xaxes(title_text="Age", row=1, col=1)
    fig.update_yaxes(title_text="Max Heart Rate", row=1, col=1)
    
    # [ROW 1, COL 2] RELIABILITY CHECK (Confusion Matrix for Heart Disease)
    try:
        if st.session_state.multi_target_model is not None and hasattr(st.session_state.multi_target_model, 'estimators_') and len(st.session_state.multi_target_model.estimators_) > 0:
            y_pred_heart = st.session_state.multi_target_model.estimators_[0].predict(st.session_state.X_test)
            cm = confusion_matrix(st.session_state.y_test['Has_HeartDisease'], y_pred_heart)
            
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=['Healthy', 'Sick'],
                    y=['Healthy', 'Sick'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate="%{text}",
                    showscale=False
                ),
                row=1, col=2
            )
    except (IndexError, AttributeError, TypeError):
        pass
    
    # [ROW 1, COL 3] CONFIDENCE (Accuracy Bars)
    if st.session_state.multi_target_model is not None and hasattr(st.session_state.multi_target_model, 'predict'):
        try:
            y_pred = st.session_state.multi_target_model.predict(st.session_state.X_test)
            accuracies = [accuracy_score(st.session_state.y_test[col], y_pred[:, i]) for i, col in enumerate(st.session_state.y_test.columns)]
            
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=accuracies,
                    marker_color=['#3498db', '#e74c3c', '#f1c40f']
                ),
                row=1, col=3
            )
            
            # Add accuracy values on bars
            for i, acc in enumerate(accuracies):
                fig.add_annotation(
                    x=i,
                    y=acc + 0.02,
                    text=f"{acc:.1%}",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    row=1, col=3
                )
                
            fig.update_yaxes(range=[0, 1.15], row=1, col=3)
        except Exception:
            pass
    
    # [ROW 2, COL 1] EXPLAINABILITY (Feature Importance)
    try:
        if st.session_state.multi_target_model is not None and hasattr(st.session_state.multi_target_model, 'estimators_'):
            importances = np.mean([est.feature_importances_ for est in st.session_state.multi_target_model.estimators_], axis=0)
            indices = np.argsort(importances)
            
            fig.add_trace(
                go.Bar(
                    x=importances[indices],
                    y=[st.session_state.X_train.columns[i] for i in indices],
                    orientation='h',
                    marker_color='#2ecc71'
                ),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Impact Score", row=2, col=1)
    except (AttributeError, IndexError, TypeError):
        pass
    
    # [ROW 2, COL 2] RISK DISTRIBUTION
    risk_counts = st.session_state.df[['Has_HeartDisease', 'Has_Hypertension', 'Has_Diabetes_Risk']].sum()
    
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=risk_counts.values,
            marker_colors=['#3498db', '#e74c3c', '#f1c40f']
        ),
        row=2, col=2
    )
    
    # [ROW 2, COL 3] PATIENT RISK PROFILE
    risk_levels = ['Low', 'Medium', 'High']
    patient_risks = [preds[0], preds[1], preds[2]]
    colors = ['#27ae60' if r == 0 else '#e74c3c' for r in patient_risks]
    
    fig.add_trace(
        go.Bar(
            x=[0.5, 0.5, 0.5],
            y=risk_levels,
            orientation='h',
            marker_color='lightgray',
            width=0.3,
            showlegend=False
        ),
        row=2, col=3
    )
    
    fig.add_trace(
        go.Bar(
            x=patient_risks,
            y=risk_levels,
            orientation='h',
            marker_color=colors,
            width=0.3,
            showlegend=False
        ),
        row=2, col=3
    )
    
    # Add text labels
    for i, (risk, color) in enumerate(zip(patient_risks, colors)):
        fig.add_annotation(
            x=0.5,
            y=i,
            text='DETECTED' if risk == 1 else 'NOT DETECTED',
            showarrow=False,
            font=dict(size=12, color='white' if risk == 1 else 'black'),
            row=2, col=3
        )
    
    fig.update_xaxes(range=[0, 1], row=2, col=3)
    fig.update_xaxes(title_text="Risk Level", row=2, col=3)
    
    # Update layout
    fig.update_layout(
        title_text=f"Medical Insight Dashboard | Patient Age: {int(p_age)} | Sex: {'M' if p_sex==1 else 'F'}",
        title_x=0.5,
        height=800,
        showlegend=True
    )
    
    return fig

# Main app
def main():
    st.title("🏥 Medical Diagnosis Prediction System")
    st.markdown("---")
    
    # Initialize model
    with st.spinner("Initializing machine learning models..."):
        initialize_model()
    
    # Sidebar
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the mode",
        ["Patient Entry", "Dashboard", "History"]
    )
    
    if app_mode == "Patient Entry":
        st.header("🩺 Patient Clinical Entry Portal")
        
        # Create form
        with st.form("patient_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                p_age = st.number_input("Age", min_value=1, max_value=120, value=50)
                p_sex = st.selectbox("Sex", ["Male", "Female"])
                p_bp = st.number_input("Blood Pressure (mmHg)", min_value=50, max_value=300, value=120)
                p_chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
            
            with col2:
                p_hr = st.number_input("Max Heart Rate", min_value=40, max_value=250, value=150)
                p_fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
                
            submitted = st.form_submit_button("🔮 Predict Risk")
        
        if submitted:
            # Process prediction
            p_sex_num = 1 if p_sex == "Male" else 0
            p_fbs_num = 1 if p_fbs == "Yes" else 0
            
            input_data = pd.DataFrame([[p_age, p_sex_num, p_bp, p_chol, p_hr, p_fbs_num]], 
                                    columns=st.session_state.feature_cols)
            filled = pd.DataFrame(st.session_state.imputer.transform(input_data), 
                                columns=st.session_state.feature_cols)
            scaled = st.session_state.scaler.transform(filled)
            
            cluster_id = st.session_state.kmeans.predict(scaled)[0]
            filled['Patient_Profile_Cluster'] = cluster_id
            preds = st.session_state.multi_target_model.predict(filled)[0]
            
            # Save prediction
            patient_data = {
                'age': p_age,
                'sex': p_sex,
                'blood_pressure': p_bp,
                'cholesterol': p_chol,
                'max_heart_rate': p_hr,
                'fasting_blood_sugar_high': p_fbs
            }
            save_prediction(patient_data, preds, cluster_id)
            
            # Store results in session state
            st.session_state.results = {
                'patient_data': patient_data,
                'cluster_id': int(cluster_id),
                'predictions': {
                    'heart_disease': bool(preds[0]),
                    'hypertension': bool(preds[1]),
                    'diabetes_risk': bool(preds[2])
                }
            }
            
            # Show results
            st.success("Prediction completed successfully!")
            st.subheader("📋 Diagnostic Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Patient Information")
                st.write(f"**Age:** {patient_data['age']} years")
                st.write(f"**Sex:** {patient_data['sex']}")
                st.write(f"**Blood Pressure:** {patient_data['blood_pressure']} mmHg")
                st.write(f"**Cholesterol:** {patient_data['cholesterol']} mg/dl")
                st.write(f"**Max Heart Rate:** {patient_data['max_heart_rate']}")
                st.write(f"**High Fasting Blood Sugar:** {patient_data['fasting_blood_sugar_high']}")
            
            with col2:
                st.markdown("### Risk Assessment Results")
                st.write(f"**Risk Profile Group:** Cluster {cluster_id}")
                
                # Heart Disease
                if preds[0] == 1:
                    st.error("Heart Disease: POSITIVE (Risk Detected)")
                else:
                    st.success("Heart Disease: Negative (Healthy)")
                
                # Hypertension
                if preds[1] == 1:
                    st.error("Hypertension: POSITIVE (Risk Detected)")
                else:
                    st.success("Hypertension: Negative (Healthy)")
                
                # Diabetes Risk
                if preds[2] == 1:
                    st.error("Diabetes Risk: POSITIVE (Risk Detected)")
                else:
                    st.success("Diabetes Risk: Negative (Healthy)")
            
            # Create and display dashboard
            st.subheader("📊 Comprehensive Medical Insight Dashboard")
            with st.spinner("Generating dashboard..."):
                fig = create_dashboard(p_age, p_sex_num, p_bp, p_chol, p_hr, p_fbs_num, cluster_id, preds)
                st.plotly_chart(fig, use_container_width=True)
    
    elif app_mode == "Dashboard":
        st.header("📊 System Dashboard")
        
        if st.session_state.initialized and st.session_state.df is not None:
            # Overall statistics
            col1, col2, col3 = st.columns(3)
            
            total_patients = len(st.session_state.df)
            heart_disease_patients = st.session_state.df['Has_HeartDisease'].sum()
            hypertension_patients = st.session_state.df['Has_Hypertension'].sum()
            diabetes_patients = st.session_state.df['Has_Diabetes_Risk'].sum()
            
            col1.metric("Total Patients", total_patients)
            col2.metric("Heart Disease Cases", heart_disease_patients, 
                       f"{heart_disease_patients/total_patients*100:.1f}%")
            col3.metric("Hypertension Cases", hypertension_patients,
                       f"{hypertension_patients/total_patients*100:.1f}%")
            
            col1, col2 = st.columns(2)
            
            # Age distribution
            with col1:
                st.subheader("Age Distribution")
                fig_age = px.histogram(st.session_state.df, x="Age", nbins=20)
                st.plotly_chart(fig_age, use_container_width=True)
            
            # Risk profile distribution
            with col2:
                st.subheader("Risk Profile Distribution")
                cluster_counts = st.session_state.df['Patient_Profile_Cluster'].value_counts()
                fig_cluster = px.pie(values=cluster_counts.values, names=[f"Cluster {i}" for i in cluster_counts.index])
                st.plotly_chart(fig_cluster, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Feature Correlation")
            corr_matrix = st.session_state.df[st.session_state.feature_cols].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Please initialize the system first by entering patient data.")
    
    elif app_mode == "History":
        st.header("📋 Prediction History")
        
        if os.path.exists('patient_history.json'):
            try:
                with open('patient_history.json', 'r') as f:
                    history = json.load(f)
                
                if history:
                    # Convert to DataFrame for better display
                    history_df = pd.DataFrame(history)
                    
                    # Display recent predictions
                    st.subheader(f"Recent Predictions ({len(history)} records)")
                    
                    # Show in reverse chronological order
                    for i, record in enumerate(reversed(history[-10:])):  # Show last 10
                        with st.expander(f"Prediction {len(history)-i} - {record['timestamp'][:19].replace('T', ' ')}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### Patient Data")
                                for key, value in record['patient_data'].items():
                                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                            
                            with col2:
                                st.markdown("### Predictions")
                                preds = record['predictions']
                                st.write(f"**Risk Profile Group:** Cluster {record['cluster_id']}")
                                st.write(f"**Heart Disease:** {'POSITIVE' if preds['heart_disease'] else 'Negative'}")
                                st.write(f"**Hypertension:** {'POSITIVE' if preds['hypertension'] else 'Negative'}")
                                st.write(f"**Diabetes Risk:** {'POSITIVE' if preds['diabetes_risk'] else 'Negative'}")
                else:
                    st.info("No prediction history available yet.")
            except Exception as e:
                st.error(f"Error loading history: {e}")
        else:
            st.info("No prediction history file found. Make some predictions first!")

if __name__ == "__main__":
    main()