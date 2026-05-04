# Medical Diagnosis Prediction System

This is an enhanced medical diagnosis prediction system that uses machine learning to predict heart disease, hypertension, and diabetes risk based on patient data. The system combines powerful data analysis with an intuitive user interface to help medical professionals quickly assess patient risk profiles.

https://sicmain.streamlit.app/

##  Key Features

###  Risk Profile Clustering
- Groups patients into different risk profiles using unsupervised learning (K-Means clustering)
- Helps identify patterns and similarities between patients
- Visualizes where a patient fits within the broader population

###  Multi-Disease Prediction
- Simultaneously predicts three major health conditions:
  - Heart Disease
  - Hypertension (High Blood Pressure)
  - Diabetes Risk
- Uses advanced Random Forest algorithms for accurate predictions


<img width="1855" height="745" alt="PatientEntry" src="https://github.com/user-attachments/assets/4e2114c1-90b3-4a59-bb82-71dc9d00c56a" />


###  Interactive Dashboard
- Provides comprehensive visualizations of patient data and predictions
- Shows 6 different charts for complete insight into patient health
- Easy-to-understand graphs and metrics for medical professionals


<img width="1858" height="871" alt="dashboard" src="https://github.com/user-attachments/assets/da1485da-2618-4e96-a62f-c3731db108cd" />


###  Patient History Tracking
- Automatically saves all prediction results for future reference
- Maintains a detailed history in `patient_history.json`
- Helps track patient progress over time


<img width="1852" height="771" alt="History" src="https://github.com/user-attachments/assets/290579b0-d9ff-4d5f-8e48-9373b651cecb" />


###  Multiple Interface Options
- **Command Line Interface**: For technical users who prefer keyboard input
- **Graphical User Interface**: Easy-to-use form-based interface for all users
- **Web Application (Flask)**: Browser-based interface for remote access
- **Streamlit Application**: Modern web interface with interactive dashboards
- Choose your preferred method when starting the application

###  Self-Contained System
- Automatically generates sample data if no dataset is provided
- Works out-of-the-box with minimal setup
- No external dependencies required to get started

##  Enhanced Features

- **Patient History Tracking**: All predictions are automatically saved to `patient_history.json`
- **Enhanced Visualization Dashboard**: Comprehensive charts and graphs for better insights
- **Multiple Interface Options**: CLI, GUI, Flask Web, and Streamlit interfaces available
- **Automatic Data Generation**: Creates sample dataset if none exists
- **Improved Error Handling**: Better error messages and recovery mechanisms
- **Deployment Scripts**: Simplified installation and execution process

##  Requirements

- Python 3.6 or higher
- Required Python packages (automatically installed by deploy scripts):
  - pandas (data manipulation)
  - numpy (numerical computing)
  - matplotlib (plotting and visualization)
  - seaborn (statistical data visualization)
  - scikit-learn (machine learning)
  - flask (web framework)
  - flask-wtf (web forms)
  - streamlit (modern web apps)
  - plotly (interactive charts)

##  Installation & Deployment

### Quick Setup Options

#### Desktop Application
1. Run the deployment script:
   ```
   python deploy.py
   ```
   
#### Web Application (Flask)
1. Run the web deployment script:
   ```
   python web_deploy.py
   ```

#### Streamlit Application
1. Run the Streamlit deployment script:
   ```
   python streamlit_deploy.py
   ```

2. The deployment process will automatically:
   - Check your Python version
   - Install all required packages
   - Create convenient startup scripts
   - Provide clear instructions for running the application

### Manual Installation
If you prefer to install manually:
```
pip install pandas numpy matplotlib seaborn scikit-learn flask flask-wtf streamlit plotly
```

##  Running the Application

### Desktop Application

After deployment, you can run the desktop application in multiple ways:

#### Method 1: Using Startup Scripts (Easiest)
- **Windows**: Double-click `start_app.bat`
- **Linux/Mac**: Run `./start_app.sh`
  
![WhatsApp Image 2025-11-27 at 21 53 44_f85ac281](https://github.com/user-attachments/assets/a5868333-12aa-4837-86d5-02574912c0b2)

#### Method 2: Direct Execution
```
python HYbrid.py
```

### Web Application (Flask)

To run the Flask web application:

#### Method 1: Using Web Startup Scripts (Easiest)
- **Windows**: Double-click `start_web_app.bat`
- **Linux/Mac**: Run `./start_web_app.sh`

#### Method 2: Direct Execution
```
python app.py
```

Once started, access the web application at: http://localhost:5000

![WhatsApp Image 2025-11-27 at 22 00 52_cc6e69c4](https://github.com/user-attachments/assets/30f10b0d-e923-4ba9-b5d8-c60f2a3eb8cd)

### Streamlit Application

To run the Streamlit application:

#### Method 1: Using Streamlit Startup Scripts (Easiest)
- **Windows**: Double-click `start_streamlit_app.bat`
- **Linux/Mac**: Run `./start_streamlit_app.sh`

#### Method 2: Direct Execution
```
streamlit run streamlit_app.py
```

Once started, access the Streamlit application at: http://localhost:8501

##  Deploying as a Website

To deploy your medical diagnosis system as a public website:

### Prerequisites
1. Ensure you have all the required files:
   - `app.py` (main web application)
   - `Procfile` (for Heroku deployment)
   - `runtime.txt` (Python version specification)
   - `requirements.txt` (dependencies)
   - `templates/` directory with HTML files

### Deployment Options

#### Heroku (Recommended for beginners)
1. Install the Heroku CLI from https://devcenter.heroku.com/articles/heroku-cli
2. Create a Heroku account
3. Login to Heroku: `heroku login`
4. Create a new app: `heroku create your-app-name`
5. Deploy: `git push heroku main`
6. Open your app: `heroku open`

#### PythonAnywhere
1. Sign up at https://www.pythonanywhere.com/
2. Upload your files
3. Create a new web app and configure it to run your Flask application

#### Other Platforms
- AWS Elastic Beanstalk
- Google Cloud Platform
- Microsoft Azure
- DigitalOcean

![WhatsApp Image 2025-11-27 at 22 20 50_37dac132](https://github.com/user-attachments/assets/778a2662-70f2-42e0-9fc9-7919881ea865)

### Environment Variables
For production deployment, set these environment variables:
- `SECRET_KEY`: A random secret key for Flask security
- `PORT`: Port number (defaults to 5000)

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)

##  How to Use

### Desktop Application
When you start the desktop application, you'll be prompted to choose between two interfaces:

#### Command Line Interface (Option 1)
1. Enter patient information when prompted:
   - Age
   - Sex (M/F)
   - Blood Pressure
   - Cholesterol Level
   - Maximum Heart Rate
   - Fasting Blood Sugar (> 120 mg/dl)

2. Receive immediate analysis and risk predictions

#### Graphical User Interface (Option 2)
1. Fill in the patient information in the easy-to-use form
2. Click "Predict Risk" to analyze the data
3. View results in a popup window

### Web Application (Flask)
1. Open your web browser and go to http://localhost:5000
2. Fill in the patient information in the web form
3. Click "Predict Risk" to analyze the data
4. View comprehensive results and visualizations in your browser

### Streamlit Application
1. Open your web browser and go to http://localhost:8501
2. Use the sidebar to navigate between different modes:
   - **Patient Entry**: Enter patient data and get predictions
   - **Dashboard**: View system-wide statistics and visualizations
   - **History**: Review previous predictions
3. Fill in patient information in the form
4. Click "Predict Risk" to analyze the data
5. View interactive results and visualizations

##  Output & Results

The system provides comprehensive analysis including:

### Risk Profile Assignment
- Assigns the patient to one of three risk profile clusters
- Shows where the patient fits within the broader population

### Individual Risk Predictions
- Heart Disease Risk: Positive (Risk Detected) or Negative (Healthy)
- Hypertension Risk: Positive (Risk Detected) or Negative (Healthy)
- Diabetes Risk: Positive (Risk Detected) or Negative (Healthy)

### Visualization Dashboard
The system generates 6 detailed visualizations:
1. **Patient Profile Context**: Shows where the patient fits among all profiles
2. **AI Reliability Metrics**: Confusion matrix showing prediction accuracy
3. **System Confidence Levels**: Bar chart showing confidence for each prediction
4. **Key Risk Factors**: Horizontal bar chart showing most important factors
5. **Population Risk Distribution**: Pie chart showing overall risk distribution
6. **Individual Risk Assessment**: Visual representation of patient's specific risks

### Data Storage
- All predictions are automatically saved to `patient_history.json`
- Each entry includes timestamp, patient data, and prediction results

##  Project Files

- `HYbrid.py`: Main desktop application with both CLI and GUI interfaces
- `app.py`: Flask web application
- `streamlit_app.py`: Streamlit web application
- `deploy.py`: Automated deployment script for desktop app
- `web_deploy.py`: Automated deployment script for Flask web app
- `streamlit_deploy.py`: Automated deployment script for Streamlit app
- `requirements.txt`: Python package dependencies
- `start_app.bat/sh`: Platform-specific startup scripts for desktop app
- `start_web_app.bat/sh`: Platform-specific startup scripts for Flask web app
- `start_streamlit_app.bat/sh`: Platform-specific startup scripts for Streamlit app
- `heart_disease_uci.csv`: Medical dataset (automatically created if missing)
- `patient_history.json`: Saved prediction results and patient history
- `setup.bat`: Windows setup script
- `templates/`: HTML templates for Flask web application
- `Procfile`: Configuration for Heroku deployment
- `runtime.txt`: Python runtime specification
- `DEPLOYMENT.md`: Detailed deployment guide

##  Contributing

This is an open-source project designed to help medical professionals and researchers. Contributions are welcome!

You can contribute by:
- Reporting bugs or issues
- Suggesting new features or enhancements
- Submitting pull requests with improvements
- Sharing the project with others in the medical community

##  Support

If you encounter any issues or have questions about the system:
1. Check that all requirements are installed
2. Ensure you're using Python 3.6 or higher
3. Run the appropriate deploy script to reinstall dependencies
4. Contact the development team through GitHub issues

##  License

This project is open-source and available for use in medical research and practice. 
Please cite appropriately if used in academic or clinical settings.
