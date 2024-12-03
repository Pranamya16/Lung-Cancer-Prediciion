import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
# Load dataset
data_path = "surveylungcancer.csv"
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    st.error("Dataset not found in the specified folder.")
    st.stop()

# Clean column names (remove spaces, standardize case)
data.columns = data.columns.str.strip().str.upper().str.replace(" ", "_")

# Convert target variable to numeric
data["LUNG_CANCER"] = (data["LUNG_CANCER"] == "YES").astype(int)

# Convert GENDER to numeric (M=1, F=0)
data["GENDER"] = (data["GENDER"] == "M").astype(int)

# Create sidebar for user input
st.sidebar.title("Patient Information")

# Input fields in sidebar
gender = st.sidebar.selectbox("Gender", options=["M", "F"])
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=50)

# Binary features with Yes/No selection
smoking = st.sidebar.selectbox("Smoking", options=["No", "Yes"])
yellow_fingers = st.sidebar.selectbox("Yellow Fingers", options=["No", "Yes"])
anxiety = st.sidebar.selectbox("Anxiety", options=["No", "Yes"])
peer_pressure = st.sidebar.selectbox("Peer Pressure", options=["No", "Yes"])
chronic_disease = st.sidebar.selectbox("Chronic Disease", options=["No", "Yes"])
fatigue = st.sidebar.selectbox("Fatigue", options=["No", "Yes"])
allergy = st.sidebar.selectbox("Allergy", options=["No", "Yes"])
wheezing = st.sidebar.selectbox("Wheezing", options=["No", "Yes"])
alcohol_consuming = st.sidebar.selectbox("Alcohol Consuming", options=["No", "Yes"])
coughing = st.sidebar.selectbox("Coughing", options=["No", "Yes"])
shortness_of_breath = st.sidebar.selectbox("Shortness of Breath", options=["No", "Yes"])
swallowing_difficulty = st.sidebar.selectbox(
    "Swallowing Difficulty", options=["No", "Yes"]
)
chest_pain = st.sidebar.selectbox("Chest Pain", options=["No", "Yes"])

# Main content area title
st.title("Lung Cancer Prediction")

# Convert input to DataFrame with correct numeric values
input_data = pd.DataFrame(
    {
        "GENDER": [1 if gender == "M" else 0],
        "AGE": [age],
        "SMOKING": [2 if smoking == "Yes" else 1],
        "YELLOW_FINGERS": [2 if yellow_fingers == "Yes" else 1],
        "ANXIETY": [2 if anxiety == "Yes" else 1],
        "PEER_PRESSURE": [2 if peer_pressure == "Yes" else 1],
        "CHRONIC_DISEASE": [2 if chronic_disease == "Yes" else 1],
        "FATIGUE": [2 if fatigue == "Yes" else 1],
        "ALLERGY": [2 if allergy == "Yes" else 1],
        "WHEEZING": [2 if wheezing == "Yes" else 1],
        "ALCOHOL_CONSUMING": [2 if alcohol_consuming == "Yes" else 1],
        "COUGHING": [2 if coughing == "Yes" else 1],
        "SHORTNESS_OF_BREATH": [2 if shortness_of_breath == "Yes" else 1],
        "SWALLOWING_DIFFICULTY": [2 if swallowing_difficulty == "Yes" else 1],
        "CHEST_PAIN": [2 if chest_pain == "Yes" else 1],
    }
)

# Train the models
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train models
@st.cache_resource
def train_models():
    lr_model = LogisticRegression(max_iter=1000)
    dt_model = xgb.XGBClassifier()
    hybrid_model = VotingClassifier(
        estimators=[("lr", lr_model), ("dt", dt_model)], voting="soft"
    )

    lr_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    hybrid_model.fit(X_train, y_train)

    return lr_model, dt_model, hybrid_model


lr_model, dt_model, hybrid_model = train_models()

# Make predictions
lr_pred = lr_model.predict_proba(input_data)[0][1]
dt_pred = dt_model.predict_proba(input_data)[0][1]
hybrid_pred = hybrid_model.predict_proba(input_data)[0][1]

# Display results
st.write("### Prediction Results")

# Overall prediction
final_prediction = (
    "Critical Risk"
    if hybrid_pred > 0.75
    else "High Risk"
    if hybrid_pred > 0.5
    else "Moderate Risk"
    if hybrid_pred > 0.25
    else "Low Risk"
)
st.write(f"### Overall Assessment: {final_prediction}")
st.write(f"Probability of lung cancer: {hybrid_pred:.2%}")

# Display model performance metrics
st.write("### Model Performance Metrics")
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

with metrics_col1:
    lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
    st.metric("Logistic Regression Accuracy", f"{lr_accuracy:.2%}")

with metrics_col2:
    dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))
    st.metric("XGBoost Accuracy", f"{dt_accuracy:.2%}")

with metrics_col3:
    hybrid_accuracy = accuracy_score(y_test, hybrid_model.predict(X_test))
    st.metric("Hybrid Model Accuracy", f"{hybrid_accuracy:.2%}")

# Add Data Visualization
st.write("### Model Fit Visualization")

# Get predictions for all data points
X_all_pred_lr = lr_model.predict_proba(X)[:, 1]
X_all_pred_dt = dt_model.predict_proba(X)[:, 1]
X_all_pred_hybrid = hybrid_model.predict_proba(X)[:, 1]

# Create visualization columns
viz_col1, viz_col2, viz_col3 = st.columns(3)

with viz_col1:
    st.write("Logistic Regression Fit")
    fig_lr = plt.figure(figsize=(6, 4))
    sns.regplot(
        x=X_all_pred_lr, y=y, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"}
    )
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Outcome")
    plt.title("Logistic Regression Fit")
    st.pyplot(fig_lr)
    plt.close()

with viz_col2:
    st.write("XGBoost Fit")
    fig_dt = plt.figure(figsize=(6, 4))
    sns.regplot(
        x=X_all_pred_dt, y=y, scatter_kws={"alpha": 0.5}, line_kws={"color": "green"}
    )
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Outcome")
    plt.title("XGBoost Fit")
    st.pyplot(fig_dt)
    plt.close()

with viz_col3:
    st.write("Hybrid Model Fit")
    fig_hybrid = plt.figure(figsize=(6, 4))
    sns.regplot(
        x=X_all_pred_hybrid, y=y, scatter_kws={"alpha": 0.5}, line_kws={"color": "blue"}
    )
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Outcome")
    plt.title("Hybrid Model Fit")
    st.pyplot(fig_hybrid)
    plt.close()

# Add additional visualization for feature importance
st.write("### Feature Importance")

# Get feature importance from XGBoost
dt_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": dt_model.feature_importances_}
).sort_values("Importance", ascending=False)

# Plot feature importance
fig_importance = plt.figure(figsize=(12, 6))
sns.barplot(data=dt_importance, x="Importance", y="Feature")
plt.title("Feature Importance from XGBoost Model")
st.pyplot(fig_importance)
plt.close()

# Add correlation heatmap
st.write("### Feature Correlation Heatmap")
fig_corr = plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
st.pyplot(fig_corr)
plt.close()

# Distribution of predictions
st.write("### Distribution of Predictions")
fig_dist = plt.figure(figsize=(12, 6))
plt.hist(X_all_pred_lr, alpha=0.5, label="Logistic Regression", bins=20)
plt.hist(X_all_pred_dt, alpha=0.5, label="XGBoost", bins=20)
plt.hist(X_all_pred_hybrid, alpha=0.5, label="Hybrid Model", bins=20)
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Distribution of Predictions by Model")
plt.legend()
st.pyplot(fig_dist)
plt.close()
