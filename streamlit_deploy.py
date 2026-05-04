#!/usr/bin/env python3
"""
Deployment script for the Medical Diagnosis Prediction System Streamlit Application
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python 3.6+ is installed"""
    if sys.version_info < (3, 6):
        print("Python 3.6 or higher is required.")
        return False
    return True

def install_requirements():
    """Install required packages"""
    if os.path.exists('requirements.txt'):
        print("Installing required packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("Required packages installed successfully.")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install required packages.")
            return False
    else:
        print("requirements.txt not found.")
        return False

def run_streamlit_application():
    """Run the Streamlit application"""
    if os.path.exists('streamlit_app.py'):
        print("Starting the Medical Diagnosis Prediction System Streamlit Application...")
        print("Access the application at: http://localhost:8501")
        try:
            subprocess.check_call([sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'])
        except subprocess.CalledProcessError:
            print("Failed to start the Streamlit application.")
    else:
        print("Main application file (streamlit_app.py) not found.")

def create_streamlit_startup_script():
    """Create a startup script for easy execution"""
    system = platform.system()
    
    if system == "Windows":
        # Create a batch file for Windows
        with open("start_streamlit_app.bat", "w") as f:
            f.write("@echo off\n")
            f.write("python -m streamlit run streamlit_app.py\n")
            f.write("pause\n")
        print("Created start_streamlit_app.bat for Windows")
        
    else:
        # Create a shell script for Linux/Mac
        with open("start_streamlit_app.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("python3 -m streamlit run streamlit_app.py\n")
        # Make it executable
        os.chmod("start_streamlit_app.sh", 0o755)
        print("Created start_streamlit_app.sh for Linux/Mac")

def main():
    """Main deployment function"""
    print("Medical Diagnosis Prediction System Streamlit Application Deployment")
    print("=" * 70)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    print("\nStep 1: Installing dependencies...")
    if not install_requirements():
        print("Warning: Could not install all dependencies. Trying to continue...")
    
    # Create startup script
    print("\nStep 2: Creating startup script...")
    create_streamlit_startup_script()
    
    # Provide instructions
    print("\nDeployment completed!")
    print("\nTo run the Streamlit application:")
    if platform.system() == "Windows":
        print("  Double-click start_streamlit_app.bat")
        print("  OR run: streamlit run streamlit_app.py")
    else:
        print("  Run: ./start_streamlit_app.sh")
        print("  OR run: streamlit run streamlit_app.py")
    
    print("\nOnce started, access the application at: http://localhost:8501")
    
    # Ask if user wants to run the app now
    choice = input("\nDo you want to run the Streamlit application now? (y/n): ").strip().lower()
    if choice in ['y', 'yes']:
        run_streamlit_application()

if __name__ == "__main__":
    main()