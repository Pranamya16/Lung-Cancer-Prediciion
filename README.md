# Lung Cancer Prediction Application

## 🩺 Overview
This Streamlit-powered machine learning application provides an advanced lung cancer risk assessment tool. By leveraging multiple sophisticated classification models, the application analyzes various patient health parameters to estimate lung cancer probability.

Click here to try: https://lung-cancer-prediction-by-pranamya.streamlit.app

## 🚀 Features

### Predictive Models
- **Logistic Regression**: A classic statistical modeling technique
- **XGBoost**: An advanced gradient boosting algorithm
- **Hybrid Voting Classifier**: Combines multiple models for enhanced accuracy

### Comprehensive Analysis
- Interactive risk prediction interface
- Detailed model performance metrics
- Advanced data visualizations
  - Feature importance charts
  - Correlation heatmaps
  - Model fit plots
  - Prediction distribution analysis

## ⚙️ Requirements
- Python 3.8+
- Libraries listed in `requirements.txt`

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Pranamya16/Lung-Cancer-Prediction.git
cd Lung-Cancer-Prediction
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🏃 Running the Application
```bash
streamlit run prediction.py
```

## 📊 Risk Categories
The application classifies lung cancer risk into four categories:
- **Critical Risk**: > 75% probability
- **High Risk**: > 50% probability
- **Moderate Risk**: > 25% probability
- **Low Risk**: ≤ 25% probability

## 🔍 Input Features
The model considers 15 critical health parameters:
1. Gender
2. Age
3. Smoking status
4. Yellow fingers
5. Anxiety levels
6. Peer pressure
7. Chronic disease
8. Fatigue
9. Allergies
10. Wheezing
11. Alcohol consumption
12. Coughing
13. Shortness of breath
14. Swallowing difficulty
15. Chest pain

## 📈 Visualizations
- Model Fit Plots
- Feature Importance Graph
- Feature Correlation Heatmap
- Prediction Distribution Chart

## ⚠️ Medical Disclaimer
**Important**: This application is a predictive tool for educational and screening purposes. It is not a substitute for professional medical diagnosis. Always consult healthcare professionals for accurate medical advice and comprehensive evaluation.

## 👥 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.



## 👨‍💻 Author
Pranamya Deshpande
