# 📉 Customer Churn Prediction System

An end-to-end machine learning application that predicts customer churn using telecom data.  
This project was developed as part of an MSc Data Analytics dissertation.

---

## 🚀 Project Overview

Customer churn prediction is critical for businesses to retain customers and reduce revenue loss.  
This system uses machine learning models to identify customers at risk of leaving.

---

## 🎯 Features

- ✔ Data preprocessing and feature engineering  
- ✔ Multiple ML models (Logistic Regression, Decision Tree, Random Forest)  
- ✔ Model evaluation (Accuracy, Precision, Recall, F1-score, ROC-AUC)  
- ✔ Best model selection  
- ✔ Interactive web application (Streamlit)  
- ✔ Real-time churn prediction  

---

## 🧠 Models Used

| Model                | Description |
|---------------------|------------|
| Logistic Regression | Best performing model (selected) |
| Decision Tree       | Baseline model |
| Random Forest       | Ensemble model |

---

## 📊 Model Performance

The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

👉 Logistic Regression achieved the best balance and was selected for deployment.

---

## 🖥️ Application Interface

The Streamlit application allows users to:
- Input customer details  
- Predict churn probability  
- View risk level (Low / Medium / High)  
- Understand model interpretation  

---

##  Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

### 2. Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run the application
streamlit run app/app.py

👉 The application will open in your browser at:

http://localhost:8501

customer-churn-prediction/
│
├── data/                # Dataset
├── models/              # Trained models (.pkl files)
├── notebooks/           # Jupyter notebooks (EDA & experiments)
├── src/                 # Preprocessing, training, prediction scripts
├── app/                 # Streamlit application
├── README.md
└── requirements.txt

Key Insights
Month-to-month contracts increase churn risk
High monthly charges correlate with higher churn
Lack of technical support increases likelihood of churn
Customer commitment and service quality are key retention factors

Academic Context

This project was developed as part of an MSc Data Analytics dissertation focusing on:

Customer churn analysis
Machine learning model comparison
Predictive analytics
Business decision support

Authur
Munamalpe Liyanage Thilan Awantha Liyanage