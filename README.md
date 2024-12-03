# Lung-Cancer-Prediciion
Predict the Patient has Lung Cancer probability or not using Hybrid Model(LogisticRegression and XGBoost)
# Lung Cancer Prediction Application

## Overview
This is a Streamlit-based machine learning application that predicts lung cancer risk using multiple classification models.

## Requirements
- Python 3.8+
- Libraries:
  - streamlit
  - pandas
  - scikit-learn
  - xgboost
  - seaborn
  - matplotlib

## Installation
1. Clone the repository
2. Install required dependencies:
```bash
pip install streamlit pandas scikit-learn xgboost seaborn matplotlib
```

## Project Structure
- `prediction.py`: Main Streamlit application script
- `surveylungcancer.csv`: Dataset containing lung cancer survey data

## Features
- Interactive web interface for lung cancer risk prediction
- Multiple machine learning models:
  - Logistic Regression
  - XGBoost
  - Hybrid Voting Classifier
- Comprehensive model performance metrics
- Feature importance visualization
- Correlation heatmap
- Prediction distribution charts

## How to Run
```bash
streamlit run prediction.py
```

## Model Performance
The application uses three models:
- Logistic Regression
- XGBoost
- Hybrid Voting Classifier

Each model's accuracy is displayed in the application.

## Input Features
- Gender
- Age
- Smoking status
- Yellow fingers
- Anxiety
- Peer pressure
- Chronic disease
- Fatigue
- Allergy
- Wheezing
- Alcohol consumption
- Coughing
- Shortness of breath
- Swallowing difficulty
- Chest pain

## Risk Categories
- Critical Risk: Probability > 75%
- High Risk: Probability > 50%
- Moderate Risk: Probability > 25%
- Low Risk: Probability ≤ 25%

## Visualizations
1. Model Fit Plots
2. Feature Importance
3. Feature Correlation Heatmap
4. Prediction Distribution

## Disclaimer
This is a predictive tool and should not replace professional medical advice. Always consult healthcare professionals for accurate diagnosis.

## Authors
Pranamya Deshpande
