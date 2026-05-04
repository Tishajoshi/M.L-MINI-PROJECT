# Deployment Guide for Medical Diagnosis Prediction System

This guide will help you deploy your medical diagnosis prediction system as a website using various platforms.

## Prerequisites

Before deploying, ensure you have:
1. Created the following files in your project:
   - `app.py` (Flask application)
   - `Procfile` (for Heroku)
   - `runtime.txt` (for Heroku)
   - `requirements.txt` (dependencies)
   - `templates/` directory with HTML files

## Deployment Options

### Option 1: Heroku (Recommended for beginners)

1. **Install Heroku CLI**
   - Download from https://devcenter.heroku.com/articles/heroku-cli
   - Install and log in to your Heroku account

2. **Prepare your application**
   ```bash
   # Initialize git repository (if not already done)
   git init
   git add .
   git commit -m "Prepare for Heroku deployment"
   ```

3. **Create and deploy to Heroku**
   ```bash
   # Create Heroku app
   heroku create your-medical-app-name
   
   # Set Python buildpack
   heroku buildpacks:set heroku/python
   
   # Deploy your app
   git push heroku main
   ```

4. **Open your deployed application**
   ```bash
   heroku open
   ```

### Option 2: PythonAnywhere

1. **Sign up** at https://www.pythonanywhere.com/
2. **Upload your files** using the web interface or Git
3. **Create a new web app**:
   - Go to the "Web" tab
   - Click "Add a new web app"
   - Choose "Flask" and your Python version
   - Select "Manual configuration"
   - Edit the WSGI configuration file to point to your `app.py`

### Option 3: AWS Elastic Beanstalk

1. **Install EB CLI**
   ```bash
   pip install awsebcli
   ```

2. **Initialize and deploy**
   ```bash
   eb init
   eb create medical-diagnosis-app
   eb open
   ```

### Option 4: Google Cloud Platform

1. **Install Google Cloud SDK**
2. **Create an `app.yaml` file**:
   ```yaml
   runtime: python39
   ```

3. **Deploy**:
   ```bash
   gcloud app deploy
   ```

## Environment Variables

For production, set these environment variables:
- `SECRET_KEY`: A random secret key for Flask
- `PORT`: Port number (defaults to 5000)

## Troubleshooting

### Common Issues

1. **Module not found errors**:
   - Ensure all dependencies are in `requirements.txt`
   - Check that you're using the correct Python version

2. **Application crashes on startup**:
   - Check logs: `heroku logs --tail` (for Heroku)
   - Ensure `Procfile` is correctly formatted

3. **Blank pages or 500 errors**:
   - Check that template files are in the `templates/` directory
   - Verify file permissions

## Scaling for Production

For a production deployment, consider:

1. **Database**: Replace file-based storage with a proper database
2. **Caching**: Implement caching for better performance
3. **Security**: Use HTTPS and proper authentication
4. **Monitoring**: Add logging and monitoring tools
5. **Load balancing**: For high-traffic applications

## Updating Your Deployed Application

To update your deployed application:

1. **Make changes to your code**
2. **Commit changes**:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

3. **Deploy updates**:
   ```bash
   # For Heroku
   git push heroku main
   
   # For other platforms, follow their specific deployment process
   ```

## Support

If you encounter issues:
1. Check the platform's documentation
2. Review application logs
3. Ensure all dependencies are correctly specified
4. Verify environment variables are set correctly