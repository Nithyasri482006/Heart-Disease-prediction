Heart Disease Prediction System

Project Overview

This project is a Machine Learning-based Heart Disease Prediction System developed using Python, Scikit-learn, and Streamlit. The application predicts whether a person is at risk of heart disease based on health-related parameters such as age, blood pressure, cholesterol, glucose level, smoking habits, alcohol consumption, physical activity, height, and weight.

Features

- Data preprocessing and normalization
- Exploratory Data Analysis (EDA)
- Multiple model comparison
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Tuned Random Forest
- Model evaluation using Accuracy, Confusion Matrix, Precision, Recall, and F1-Score
- Interactive Streamlit web application
- Real-time prediction through user input

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

Model Performance

Model| Accuracy
Logistic Regression| 72.36%
Decision Tree| 63.03%
Random Forest| 74.00%
Tuned Random Forest| 74.01%

Project Structure

- data/ - Dataset files
- models/ - Trained model and scaler files
- notebook/ - Project notebook and documentation
- app.py - Streamlit application
- requirements.txt - Required Python libraries

Conclusion

The Tuned Random Forest model achieved the highest accuracy and was selected for deployment. The Streamlit application provides a user-friendly interface for predicting heart disease risk based on user inputs.

# Heart-Disease-prediction
