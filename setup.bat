@echo off
echo Medical Diagnosis Prediction System Setup
echo ========================================
echo.
echo This script will deploy and set up the medical diagnosis prediction system.
echo.
echo Press any key to continue...
pause >nul

echo.
echo Running deployment script...
python deploy.py

echo.
echo Setup complete!
echo.
echo To run the application, double-click on start_app.bat
echo Or run: python HYbrid.py
echo.
pause