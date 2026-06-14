# Telco Customer Churn Analysis

A data science project built to answer one business question: which telecom customers are about to leave, and what's driving them out?

---

## What This Project Is About

Churn is expensive. Every customer who leaves is revenue lost, and in telecom, the margins are tight enough that it matters. This project works through 7,032 customer records; demographics, services, contracts, billing, to find the patterns that separate customers who stay from those who leave, then builds a model to flag at-risk customers before they go.

---

## Live Dashboard

🔗 [View the interactive Streamlit dashboard](https://zz5ocr749xdx5nrhzgsyad.streamlit.app)

---

## What the Data Showed

- **1 in 4 customers churns** - 26.6% overall churn rate
- **Contract type matters most** - month-to-month customers leave at 42.7% vs 2.8% for two-year contracts
- **New customers are the most vulnerable** - churned customers averaged 18 months with the company vs 38 months for those who stayed
- **Higher bills push people out** - churned customers were paying around $74/month on average vs $61 for retained ones
- **Fiber optic users churn more** - higher cost tier, more alternatives available

---

## Project Structure

    telco-churn-analysis/
    ├── data/                             # Raw dataset
    ├── notebooks/
    │   ├── 01_eda.ipynb                  # Exploratory Data Analysis
    │   ├── 02_feature_engineering.ipynb  # Preparing data for modelling
    │   └── 03_modelling.ipynb            # Training and evaluating models
    ├── dashboard/
    │   └── app.py                        # Streamlit dashboard
    ├── outputs/
    │   ├── figures/                      # Saved charts
    │   └── models/                       # Saved models
    └── README.md

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 72.6% | 49.0% | 79.4% | 60.6% | 0.835 |
| Random Forest | 76.7% | 55.3% | 64.2% | 59.4% | 0.817 |

Logistic Regression came out on top - not because it had the highest accuracy, but because it caught more actual churners (Recall: 79.4%) and had a better overall AUC (0.835). In a churn context, missing a customer who's about to leave costs more than a false alarm.

---

## Stack

- **Python** - pandas, numpy, scikit-learn, matplotlib, seaborn
- **Models** - Logistic Regression, Random Forest
- **Dashboard** - Streamlit
- **Tools** - VS Code, Jupyter Notebooks

---

## Dataset

IBM Telco Customer Churn - [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

7,032 customers · 20 features · binary classification

---

## Author

**El Mbondo**
BSc Mathematics and Computer Science — JKUAT
[GitHub](https://github.com/elmbondo)
