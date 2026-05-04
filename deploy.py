#!/usr/bin/env python3
"""
Deployment script for the Medical Diagnosis Prediction System
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

def run_application():
    """Run the main application"""
    if os.path.exists('HYbrid.py'):
        print("Starting the Medical Diagnosis Prediction System...")
        try:
            subprocess.check_call([sys.executable, 'HYbrid.py'])
        except subprocess.CalledProcessError:
            print("Failed to start the application.")
    else:
        print("Main application file (HYbrid.py) not found.")

def create_startup_script():
    """Create a startup script for easy execution"""
    system = platform.system()
    
    if system == "Windows":
        # Create a batch file for Windows
        with open("start_app.bat", "w") as f:
            f.write("@echo off\n")
            f.write("python HYbrid.py\n")
            f.write("pause\n")
        print("Created start_app.bat for Windows")
        
    else:
        # Create a shell script for Linux/Mac
        with open("start_app.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("python3 HYbrid.py\n")
        # Make it executable
        os.chmod("start_app.sh", 0o755)
        print("Created start_app.sh for Linux/Mac")

def main():
    """Main deployment function"""
    print("Medical Diagnosis Prediction System Deployment")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    print("\nStep 1: Installing dependencies...")
    if not install_requirements():
        print("Warning: Could not install all dependencies. Trying to continue...")
    
    # Create startup script
    print("\nStep 2: Creating startup script...")
    create_startup_script()
    
    # Provide instructions
    print("\nDeployment completed!")
    print("\nTo run the application:")
    if platform.system() == "Windows":
        print("  Double-click start_app.bat")
        print("  OR run: python HYbrid.py")
    else:
        print("  Run: ./start_app.sh")
        print("  OR run: python3 HYbrid.py")
    
    # Ask if user wants to run the app now
    choice = input("\nDo you want to run the application now? (y/n): ").strip().lower()
    if choice in ['y', 'yes']:
        run_application()

if __name__ == "__main__":
    main()