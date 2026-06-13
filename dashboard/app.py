import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Analysis",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .stMetric { background-color: #FFFFFF; padding: 16px; border-radius: 8px;
                border-left: 4px solid #048A81; }
    .block-container { padding-top: 2rem; }
    h1 { color: #2E4057; }
    h2 { color: #048A81; }
    .churn-yes { background-color: #FFEBEE; border-left: 4px solid #E53935;
                 padding: 16px; border-radius: 8px; margin: 8px 0; }
    .churn-no  { background-color: #E8F5E9; border-left: 4px solid #43A047;
                 padding: 16px; border-radius: 8px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

# ── Load data and models ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df[df['TotalCharges'].notnull()].reset_index(drop=True)
    df.drop(columns=['customerID'], inplace=True)
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    return df

@st.cache_resource
def load_models():
    with open('outputs/models/random_forest_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('outputs/models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('outputs/models/feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return rf, scaler, feature_cols

@st.cache_data
def get_model_results():
    df = load_data()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    target_col = 'Churn'
    categorical_cols = [c for c in df.columns if c not in numerical_cols + [target_col]]
    binary_cols = [c for c in categorical_cols if df[c].nunique() == 2]
    multi_cols  = [c for c in categorical_cols if df[c].nunique() > 2]

    le = LabelEncoder()
    df_enc = df.copy()
    for col in binary_cols:
        df_enc[col] = le.fit_transform(df_enc[col])
    df_enc['Churn'] = le.fit_transform(df_enc['Churn'])
    df_enc = pd.get_dummies(df_enc, columns=multi_cols, drop_first=True)

    X = df_enc.drop(columns=['Churn'])
    y = df_enc['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols]  = scaler.transform(X_test[numerical_cols])

    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)

    results = {}
    for name, model in [('Random Forest', rf), ('Logistic Regression', lr)]:
        yp = model.predict(X_test)
        yprob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'y_pred': yp, 'y_prob': yprob,
            'accuracy':  accuracy_score(y_test, yp),
            'precision': precision_score(y_test, yp),
            'recall':    recall_score(y_test, yp),
            'f1':        f1_score(y_test, yp),
            'auc':       roc_auc_score(y_test, yprob),
        }
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)

    return results, y_test, importance_df, X_train.columns.tolist()

df = load_data()
rf_model, scaler, feature_cols = load_models()
model_results, y_test, importance_df, train_cols = get_model_results()

# ── Sidebar ─────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/signal.png", width=60)
st.sidebar.title("📡 Telco Churn")
st.sidebar.markdown("**Navigation**")
page = st.sidebar.radio("", [
    "🏠 Overview",
    "📊 EDA & Insights",
    "🤖 Model Performance",
    "🔮 Predict Churn"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.markdown(f"- {len(df):,} customers")
st.sidebar.markdown(f"- {df.shape[1]} features")
st.sidebar.markdown(f"- {(df['Churn']=='Yes').mean()*100:.1f}% churn rate")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by **El Mbondo**")
st.sidebar.markdown("IBM Telco Customer Churn Dataset")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📡 Telco Customer Churn Analysis")
    st.markdown("### Can we predict which customers are about to leave — and why?")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        churned = (df['Churn'] == 'Yes').sum()
        st.metric("Churned Customers", f"{churned:,}", delta=f"-{churned/len(df)*100:.1f}%")
    with col3:
        st.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
    with col4:
        st.metric("Avg Tenure", f"{df['tenure'].mean():.0f} months")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df['Churn'].value_counts()
        bars = ax.bar(counts.index, counts.values, color=['#4C72B0', '#DD8452'], edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=11, fontweight='bold')
            ax.set_ylabel('Number of Customers')
            ax.set_title('Customer Churn Distribution', fontweight='bold', pad=20)
            ax.set_ylim(0, max(counts.values) * 1.2)
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            sns.despine()
            st.pyplot(fig)
            plt.close()

    with col2:
        st.subheader("Churn Rate by Contract Type")
        fig, ax = plt.subplots(figsize=(5, 4))
        churn_contract = df.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100).round(1)
        bars = ax.bar(churn_contract.index, churn_contract.values,
                      color=['#DD8452', '#4C72B0', '#55A868'], edgecolor='white')
        for bar, val in zip(bars, churn_contract.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val}%', ha='center', fontsize=11, fontweight='bold')
        ax.set_ylabel('Churn Rate (%)')
        ax.set_title('Churn Rate by Contract Type', fontweight='bold')
        ax.tick_params(axis='x', rotation=10)
        sns.despine()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("🔑 Key Findings")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Contract Type is King**\nMonth-to-month customers churn at 43% vs just 3% for two-year contracts.")
    with col2:
        st.warning("**New Customers at Risk**\nChurned customers have avg tenure of 18 months vs 38 months for loyal customers.")
    with col3:
        st.error("**High Bills Drive Churn** Churned customers pay ~\\$74/month on average vs ~\\$61 for retained customers.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Insights":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Deep dive into patterns and drivers of customer churn.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Demographics", "Services & Contract", "Numerical Features"])

    with tab1:
        st.subheader("Churn Rate by Demographics")
        demo_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, col in enumerate(demo_cols):
            churn_rate = df.groupby(col)['Churn'].apply(
                lambda x: (x == 'Yes').mean() * 100).round(1)
            bars = axes[i].bar(churn_rate.index, churn_rate.values,
                               color=['#4C72B0', '#DD8452'], edgecolor='white')
            for bar, val in zip(bars, churn_rate.values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                             f'{val}%', ha='center', fontweight='bold')
            axes[i].set_title(f'Churn Rate by {col}', fontweight='bold')
            axes[i].set_ylabel('Churn Rate (%)')
            axes[i].set_ylim(0, 60)
            sns.despine(ax=axes[i])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.subheader("Churn Rate by Service & Contract")
        service_cols = ['Contract', 'InternetService', 'PaymentMethod', 'PaperlessBilling']
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes = axes.flatten()
        for i, col in enumerate(service_cols):
            churn_rate = df.groupby(col)['Churn'].apply(
                lambda x: (x == 'Yes').mean() * 100).round(1)
            bars = axes[i].bar(churn_rate.index, churn_rate.values,
                               color='#4C72B0', edgecolor='white')
            for bar, val in zip(bars, churn_rate.values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                             f'{val}%', ha='center', fontweight='bold', fontsize=9)
            axes[i].set_title(f'Churn Rate by {col}', fontweight='bold')
            axes[i].set_ylabel('Churn Rate (%)')
            axes[i].set_ylim(0, 60)
            axes[i].tick_params(axis='x', rotation=15)
            sns.despine(ax=axes[i])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.subheader("Numerical Features by Churn Status")
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, col in enumerate(num_cols):
            churned     = df[df['Churn'] == 'Yes'][col]
            not_churned = df[df['Churn'] == 'No'][col]
            axes[i].hist(not_churned, bins=30, alpha=0.6, label='No Churn', color='#4C72B0')
            axes[i].hist(churned,     bins=30, alpha=0.6, label='Churned',  color='#DD8452')
            axes[i].set_title(f'{col} Distribution', fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].legend()
            sns.despine(ax=axes[i])
        plt.suptitle('Numerical Features by Churn Status', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        col1, col2, col3 = st.columns(3)
        for col_widget, feature in zip([col1, col2, col3], num_cols):
            with col_widget:
                churned_mean     = df[df['Churn'] == 'Yes'][feature].mean()
                not_churned_mean = df[df['Churn'] == 'No'][feature].mean()
                st.metric(
                    f"Avg {feature} — Churned",
                    f"{churned_mean:.1f}",
                    delta=f"{churned_mean - not_churned_mean:+.1f} vs retained"
                )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.markdown("Comparing Logistic Regression vs Random Forest.")
    st.markdown("---")

    # Metrics table
    st.subheader("Model Comparison")
    metrics_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'Accuracy':  [f"{v['accuracy']*100:.1f}%"  for v in model_results.values()],
        'Precision': [f"{v['precision']*100:.1f}%" for v in model_results.values()],
        'Recall':    [f"{v['recall']*100:.1f}%"    for v in model_results.values()],
        'F1 Score':  [f"{v['f1']*100:.1f}%"        for v in model_results.values()],
        'ROC-AUC':   [f"{v['auc']:.3f}"            for v in model_results.values()],
    })
    st.dataframe(metrics_df.set_index('Model'), use_container_width=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrices")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (name, res) in zip(axes, model_results.items()):
            cm = confusion_matrix(y_test, res['y_pred'])
            disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])
            disp.plot(ax=ax, colorbar=False, cmap='Blues')
            ax.set_title(name, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("ROC Curves")
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = ['#4C72B0', '#DD8452']
        for (name, res), color in zip(model_results.items(), colors):
            fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
            ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=color, lw=2)
        ax.plot([0,1],[0,1], 'k--', lw=1, label='Random (AUC=0.500)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves', fontweight='bold')
        ax.legend(loc='lower right')
        sns.despine()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Top 15 Most Important Features — Random Forest")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title('Feature Importance', fontweight='bold')
    ax.set_xlabel('Importance Score')
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Churn":
    st.title("🔮 Predict Customer Churn")
    st.markdown("Enter a customer's details below to get a live churn prediction.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        gender         = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner        = st.selectbox("Partner", ["Yes", "No"])
        dependents     = st.selectbox("Dependents", ["No", "Yes"])

    with col2:
        st.subheader("Services")
        phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service= st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup   = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_prot     = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support    = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv    = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies= st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        st.subheader("Account Info")
        contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method  = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 119, 65)
        total_charges   = st.number_input("Total Charges ($)",
                                          min_value=0.0,
                                          value=float(tenure * monthly_charges))

    st.markdown("---")
    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        # Build input row matching training columns
        input_data = {
            'gender':           1 if gender == 'Male' else 0,
            'SeniorCitizen':    1 if senior_citizen == 'Yes' else 0,
            'Partner':          1 if partner == 'Yes' else 0,
            'Dependents':       1 if dependents == 'Yes' else 0,
            'tenure':           tenure,
            'PhoneService':     1 if phone_service == 'Yes' else 0,
            'PaperlessBilling': 1 if paperless == 'Yes' else 0,
            'MonthlyCharges':   monthly_charges,
            'TotalCharges':     total_charges,
            'MultipleLines_No phone service': 1 if multiple_lines == 'No phone service' else 0,
            'MultipleLines_Yes':              1 if multiple_lines == 'Yes' else 0,
            'InternetService_Fiber optic':    1 if internet_service == 'Fiber optic' else 0,
            'InternetService_No':             1 if internet_service == 'No' else 0,
            'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
            'OnlineSecurity_Yes':             1 if online_security == 'Yes' else 0,
            'OnlineBackup_No internet service':   1 if online_backup == 'No internet service' else 0,
            'OnlineBackup_Yes':               1 if online_backup == 'Yes' else 0,
            'DeviceProtection_No internet service': 1 if device_prot == 'No internet service' else 0,
            'DeviceProtection_Yes':           1 if device_prot == 'Yes' else 0,
            'TechSupport_No internet service':    1 if tech_support == 'No internet service' else 0,
            'TechSupport_Yes':                1 if tech_support == 'Yes' else 0,
            'StreamingTV_No internet service':    1 if streaming_tv == 'No internet service' else 0,
            'StreamingTV_Yes':                1 if streaming_tv == 'Yes' else 0,
            'StreamingMovies_No internet service':1 if streaming_movies == 'No internet service' else 0,
            'StreamingMovies_Yes':            1 if streaming_movies == 'Yes' else 0,
            'Contract_One year':              1 if contract == 'One year' else 0,
            'Contract_Two year':              1 if contract == 'Two year' else 0,
            'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
            'PaymentMethod_Electronic check':        1 if payment_method == 'Electronic check' else 0,
            'PaymentMethod_Mailed check':            1 if payment_method == 'Mailed check' else 0,
        }

        input_df = pd.DataFrame([input_data])[feature_cols]
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        prediction = rf_model.predict(input_df)[0]
        probability = rf_model.predict_proba(input_df)[0][1]

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if prediction == 1:
                st.markdown(f"""
                <div class="churn-yes">
                    <h2 style="color:#E53935; margin:0">⚠️ HIGH CHURN RISK</h2>
                    <p style="font-size:18px; margin:8px 0">
                        This customer has a <strong>{probability*100:.1f}%</strong> probability of churning.
                    </p>
                    <p style="color:#666; margin:0">Consider a retention offer or proactive outreach.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="churn-no">
                    <h2 style="color:#43A047; margin:0">✅ LOW CHURN RISK</h2>
                    <p style="font-size:18px; margin:8px 0">
                        This customer has only a <strong>{probability*100:.1f}%</strong> probability of churning.
                    </p>
                    <p style="color:#666; margin:0">Customer appears stable. Continue normal engagement.</p>
                </div>
                """, unsafe_allow_html=True)

            # Probability gauge
            st.markdown("### Churn Probability")
            fig, ax = plt.subplots(figsize=(6, 1.2))
            ax.barh([''], [probability], color='#E53935' if probability > 0.5 else '#43A047',
                    height=0.5)
            ax.barh([''], [1 - probability], left=[probability],
                    color='#EEEEEE', height=0.5)
            ax.set_xlim(0, 1)
            ax.axvline(x=0.5, color='gray', linestyle='--', lw=1.5)
            ax.text(probability/2, 0, f'{probability*100:.1f}%',
                    ha='center', va='center', fontweight='bold',
                    color='white', fontsize=14)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax.set_yticks([])
            sns.despine(left=True)
            st.pyplot(fig)
            plt.close()