# Disease Risk Grouping and Diagnosis Prediction Using ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
import json
import os
from datetime import datetime
# For web interface
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False

# CONFIGURATION
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# --- 1. BACKEND: SILENT ENGINE START ---
try:
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

except Exception as e:
    print(f"Error loading dataset: {e}")
    raise SystemExit

# --- NEW: Patient History Tracking ---
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
        print(f"Prediction saved to {history_file}")
    except IOError as e:
        print(f"Could not save prediction: {e}")

# --- NEW: Enhanced Visualization ---
def create_enhanced_dashboard(p_age, p_sex, p_bp, p_chol, p_hr, p_fbs, cluster_id, preds):
    """Create enhanced dashboard with more insights"""
    labels = ['Heart Disease', 'Hypertension', 'Diabetes Risk']
    
    print("\nGENERATING ENHANCED INSIGHT DASHBOARD...")
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    plt.suptitle(f"Enhanced Medical Insight Dashboard | Patient Age: {int(p_age)} | Sex: {'M' if p_sex==1 else 'F'}", 
                 fontsize=16, fontweight='bold')

    # [ROW 1, COL 1] PATIENT CONTEXT (Where do they fit?)
    sns.scatterplot(data=df, x='Age', y='MaxHeartRate', hue='Patient_Profile_Cluster',
                    palette='viridis', alpha=0.4, ax=axs[0, 0], legend='full')  # type: ignore
    axs[0, 0].scatter(p_age, p_hr, color='red', s=300, marker='*', label='THIS PATIENT', zorder=10)  # type: ignore
    axs[0, 0].set_title("1. Patient Profile Context (Cluster Analysis)", fontweight='bold')
    axs[0, 0].legend(loc='upper right')

    # [ROW 1, COL 2] RELIABILITY CHECK (Confusion Matrix for Heart Disease)
    # Simply try to access the estimator without checking length
    try:
        y_pred_heart = multi_target_model.estimators_[0].predict(X_test)  # type: ignore
        cm = confusion_matrix(y_test['Has_HeartDisease'], y_pred_heart)  # type: ignore
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0, 1], cbar=False, annot_kws={"size": 14})  # type: ignore
        axs[0, 1].set_title("2. AI Reliability (Confusion Matrix - Heart Disease)", fontweight='bold')
        axs[0, 1].set_xlabel("Predicted Diagnosis")
        axs[0, 1].set_ylabel("True Diagnosis")
        axs[0, 1].set_xticklabels(['Healthy', 'Sick'])
        axs[0, 1].set_yticklabels(['Healthy', 'Sick'])
    except (IndexError, AttributeError):
        axs[0, 1].text(0.5, 0.5, "Data not available", ha='center', va='center')
        axs[0, 1].set_title("2. AI Reliability (Confusion Matrix - Heart Disease)", fontweight='bold')

    # [ROW 1, COL 3] CONFIDENCE (Accuracy Bars)
    if hasattr(multi_target_model, 'predict'):
        y_pred = multi_target_model.predict(X_test)  # type: ignore
        accuracies = [accuracy_score(y_test[col], y_pred[:, i]) for i, col in enumerate(y.columns)]
        bars = axs[0, 2].bar(labels, accuracies, color=['#3498db', '#e74c3c', '#f1c40f'])  # type: ignore
        axs[0, 2].set_title("3. Overall System Confidence", fontweight='bold')
        axs[0, 2].set_ylim(0, 1.15)
        for i, (bar, v) in enumerate(zip(bars, accuracies)):
            axs[0, 2].text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.1%}", 
                          ha='center', fontweight='bold')

    # [ROW 2, COL 1] EXPLAINABILITY (Feature Importance)
    try:
        importances = np.mean([est.feature_importances_ for est in multi_target_model.estimators_], axis=0)  # type: ignore
        indices = np.argsort(importances)
        axs[1, 0].barh(range(len(indices)), importances[indices], color='#2ecc71', align='center')  # type: ignore
        axs[1, 0].set_yticks(range(len(indices)))
        axs[1, 0].set_yticklabels([X.columns[i] for i in indices])
        axs[1, 0].set_title("4. Top Clinical Risk Factors (AI Logic)", fontweight='bold')
        axs[1, 0].set_xlabel("Impact Score")
    except (AttributeError, IndexError):
        axs[1, 0].text(0.5, 0.5, "Data not available", ha='center', va='center')
        axs[1, 0].set_title("4. Top Clinical Risk Factors (AI Logic)", fontweight='bold')

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

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # type: ignore
    plt.show()

# --- NEW: WEB INTERFACE OPTION ---
def launch_web_interface():
    """Launch a simple web interface for the application"""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        
        def predict_risk():
            try:
                # Get values from inputs
                p_age = float(age_entry.get())
                p_sex = 1 if sex_var.get() == 'Male' else 0
                p_bp = float(bp_entry.get())
                p_chol = float(chol_entry.get())
                p_hr = float(hr_entry.get())
                p_fbs = 1 if fbs_var.get() == 'Yes' else 0
                
                # Process prediction
                input_data = pd.DataFrame(data=[[p_age, p_sex, p_bp, p_chol, p_hr, p_fbs]], columns=feature_cols)  # type: ignore
                filled = pd.DataFrame(data=imputer.transform(input_data), columns=feature_cols)  # type: ignore
                scaled = scaler.transform(filled)
                
                if hasattr(kmeans, 'predict'):
                    cluster_id = kmeans.predict(scaled)[0]  # type: ignore
                    filled['Patient_Profile_Cluster'] = cluster_id
                    if hasattr(multi_target_model, 'predict'):
                        preds = multi_target_model.predict(filled)[0]  # type: ignore
                        
                        # Show results
                        result_text = f"Risk Profile Group: {cluster_id}\n\n"
                        labels = ['Heart Disease', 'Hypertension', 'Diabetes Risk']
                        for label, result in zip(labels, preds):
                            status = "POSITIVE (Risk Detected)" if result == 1 else "Negative (Healthy)"
                            result_text += f"{label}: {status}\n"
                        
                        messagebox.showinfo("Prediction Results", result_text)
                        
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
                
            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid numerical values for all fields.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
        
        # Create main window
        root = tk.Tk()
        root.title("Medical Diagnosis Prediction System")
        root.geometry("500x400")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky="nsew")  # type: ignore
        
        # Title
        title_label = ttk.Label(main_frame, text="Clinical Entry Portal", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Input fields
        ttk.Label(main_frame, text="Age:").grid(row=1, column=0, sticky=tk.W, pady=5)
        age_entry = ttk.Entry(main_frame)
        age_entry.grid(row=1, column=1, sticky="ew", pady=5)
        
        ttk.Label(main_frame, text="Sex:").grid(row=2, column=0, sticky=tk.W, pady=5)
        sex_var = tk.StringVar()
        sex_combo = ttk.Combobox(main_frame, textvariable=sex_var, values=['Male', 'Female'], state="readonly")
        sex_combo.grid(row=2, column=1, sticky="ew", pady=5)
        sex_combo.current(0)
        
        ttk.Label(main_frame, text="Blood Pressure:").grid(row=3, column=0, sticky=tk.W, pady=5)
        bp_entry = ttk.Entry(main_frame)
        bp_entry.grid(row=3, column=1, sticky="ew", pady=5)
        
        ttk.Label(main_frame, text="Cholesterol:").grid(row=4, column=0, sticky=tk.W, pady=5)
        chol_entry = ttk.Entry(main_frame)
        chol_entry.grid(row=4, column=1, sticky="ew", pady=5)
        
        ttk.Label(main_frame, text="Max Heart Rate:").grid(row=5, column=0, sticky=tk.W, pady=5)
        hr_entry = ttk.Entry(main_frame)
        hr_entry.grid(row=5, column=1, sticky="ew", pady=5)
        
        ttk.Label(main_frame, text="Fasting Blood Sugar > 120?:").grid(row=6, column=0, sticky=tk.W, pady=5)
        fbs_var = tk.StringVar()
        fbs_combo = ttk.Combobox(main_frame, textvariable=fbs_var, values=['Yes', 'No'], state="readonly")
        fbs_combo.grid(row=6, column=1, sticky="ew", pady=5)
        fbs_combo.current(1)
        
        # Predict button
        predict_button = ttk.Button(main_frame, text="Predict Risk", command=predict_risk)
        predict_button.grid(row=7, column=0, columnspan=2, pady=20)
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        root.mainloop()
        
    except ImportError:
        print("Tkinter is not available. Please install it to use the web interface.")

# --- 2. FRONTEND: INPUT INTERFACE ---
def get_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")
            pass

def main_cli_interface():
    """Main CLI interface for the application"""
    print("\nCLINICAL ENTRY PORTAL")
    print("---------------------")

    try:
        # Inputs
        p_age = get_input("Age: ")
        s_in = input("Sex (M/F): ").strip().upper()
        p_sex = 1 if s_in.startswith('M') else 0
        p_bp = get_input("Blood Pressure: ")
        p_chol = get_input("Cholesterol: ")
        p_hr = get_input("Max Heart Rate: ")
        f_in = input("Fasting Blood Sugar > 120? (Y/N): ").strip().upper()
        p_fbs = 1 if f_in.startswith('Y') else 0

        # Processing
        input_data = pd.DataFrame(data=[[p_age, p_sex, p_bp, p_chol, p_hr, p_fbs]], columns=feature_cols)  # type: ignore
        filled = pd.DataFrame(data=imputer.transform(input_data), columns=feature_cols)  # type: ignore
        scaled = scaler.transform(filled)

        if hasattr(kmeans, 'predict'):
            cluster_id = kmeans.predict(scaled)[0]  # type: ignore
            filled['Patient_Profile_Cluster'] = cluster_id
            if hasattr(multi_target_model, 'predict'):
                preds = multi_target_model.predict(filled)[0]  # type: ignore

                # Text Report
                print("\nDIAGNOSTIC SUMMARY")
                print("------------------")
                print(f"Risk Profile Group: {cluster_id}")
                labels = ['Heart Disease', 'Hypertension', 'Diabetes Risk']
                for label, result in zip(labels, preds):
                    status = "POSITIVE (Risk Detected)" if result == 1 else "Negative (Healthy)"
                    print(f"{label:<20} : {status}")

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

                # Enhanced Visualization
                create_enhanced_dashboard(p_age, p_sex, p_bp, p_chol, p_hr, p_fbs, cluster_id, preds)

    except Exception as e:
        print(f"Processing Error: {e}")

if __name__ == "__main__":
    print("MEDICAL DIAGNOSIS PREDICTION SYSTEM")
    print("===================================")
    print("1. Command Line Interface")
    print("2. Graphical User Interface (if available)")
    
    choice = input("Select interface (1 or 2): ").strip()
    
    # Handle different input formats
    if choice == "2" or choice.lower() == "gui" or "graphic" in choice.lower():
        if TK_AVAILABLE:
            launch_web_interface()
        else:
            print("GUI is not available. Tkinter is not installed.")
            print("Falling back to Command Line Interface...")
            main_cli_interface()
    else:
        main_cli_interface()