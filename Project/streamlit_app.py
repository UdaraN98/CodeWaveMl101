import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="SHAP & LIME Explainability Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔍 SHAP & LIME Explainability Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("⚙️ Configuration")

@st.cache_data
def load_data():
    """Load the customer churn dataset"""
    try:
        data = pd.read_csv('Data/customer_churn_dataset_prepared.csv')
        return data
    except FileNotFoundError:
        st.error("❌ Dataset not found. Please ensure 'Data/customer_churn_dataset_prepared.csv' exists.")
        return None

@st.cache_resource
def train_models(X, y):
    """Train both Logistic Regression and Decision Tree models"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Train Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Calculate accuracies
    lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
    dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))
    
    return lr_model, dt_model, X_train, X_test, y_train, y_test, lr_accuracy, dt_accuracy

def create_shap_plots(model, X_sample, model_type):
    """Create SHAP plots"""
    if model_type == "Logistic Regression":
        explainer = shap.LinearExplainer(model, X_sample)
    else:  # Decision Tree
        explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(X_sample)
    
    # Handle different SHAP value formats
    if len(shap_values.shape) == 3:  # Multi-class output
        shap_values = shap_values[:, :, 1]  # Use positive class
    
    return explainer, shap_values

def create_lime_explanation(model, X_train, X_sample, feature_names, instance_idx):
    """Create LIME explanation for a specific instance"""
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=['No Churn', 'Churn'],
        discretize_continuous=True
    )
    
    exp = explainer.explain_instance(
        X_sample.iloc[instance_idx].values,
        model.predict_proba,
        num_features=len(feature_names)
    )
    
    return exp

# Load data
data = load_data()

if data is not None:
    # Display basic info about the dataset
    st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(data))
    with col2:
        st.metric("Features", len(data.columns) - 1)  # Excluding target column
    with col3:
        churn_rate = data['Churn'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col4:
        st.metric("Data Shape", f"{data.shape[0]} x {data.shape[1]}")
    
    # Data preview
    with st.expander("📋 View Dataset Sample", expanded=False):
        st.dataframe(data.head(10))
    
    # Prepare data
    X = data.drop(columns=['Churn'])
    y = data['Churn']
    
    # Train models
    lr_model, dt_model, X_train, X_test, y_train, y_test, lr_acc, dt_acc = train_models(X, y)
    
    # Model selection
    st.markdown('<h2 class="section-header">🤖 Model Selection</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Logistic Regression Accuracy", f"{lr_acc:.3f}")
    with col2:
        st.metric("Decision Tree Accuracy", f"{dt_acc:.3f}")
    
    selected_model = st.selectbox(
        "Choose Model for Explanation:",
        ["Logistic Regression", "Decision Tree"],
        index=0
    )
    
    model = lr_model if selected_model == "Logistic Regression" else dt_model
    
    # Sample size selection
    sample_size = st.sidebar.slider("Sample Size for Analysis", 100, min(1000, len(X_test)), 500)
    X_sample = X_test.iloc[:sample_size]
    
    # SHAP Analysis
    st.markdown('<h2 class="section-header">🔎 SHAP Analysis</h2>', unsafe_allow_html=True)
    
    if st.button("Generate SHAP Plots", type="primary"):
        with st.spinner("Generating SHAP explanations..."):
            try:
                explainer, shap_values = create_shap_plots(model, X_sample, selected_model)
                
                # Summary plot
                st.subheader("SHAP Summary Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, show=False)
                st.pyplot(fig)
                plt.close()
                
                # Feature importance
                st.subheader("SHAP Feature Importance")
                feature_importance = np.mean(np.abs(shap_values), axis=0)
                importance_df = pd.DataFrame({
                    'Feature': X_sample.columns,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=True)
                
                fig_bar = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Mean Absolute SHAP Values',
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                fig_bar.update_layout(height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating SHAP plots: {str(e)}")
    
    # LIME Analysis
    st.markdown('<h2 class="section-header">🍋 LIME Analysis</h2>', unsafe_allow_html=True)
    
    # Instance selection for LIME
    instance_idx = st.selectbox(
        "Select Instance Index for LIME Explanation:",
        range(min(100, len(X_sample))),
        index=0
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Selected Instance")
        instance_data = X_sample.iloc[instance_idx]
        st.write("**Feature Values:**")
        for feature, value in instance_data.items():
            st.write(f"- **{feature}**: {value:.3f}")
        
        # Prediction for this instance
        prediction = model.predict([instance_data.values])[0]
        prediction_proba = model.predict_proba([instance_data.values])[0]
        st.write(f"**Prediction**: {'Churn' if prediction == 1 else 'No Churn'}")
        st.write(f"**Probability**: {prediction_proba[1]:.3f}")
    
    with col2:
        if st.button("Generate LIME Explanation", type="primary"):
            with st.spinner("Generating LIME explanation..."):
                try:
                    exp = create_lime_explanation(model, X_train, X_sample, X.columns.tolist(), instance_idx)
                    
                    # Get explanation as HTML
                    html_exp = exp.as_html(show_table=True)
                    
                    # Display LIME explanation
                    st.subheader("LIME Explanation")
                    st.components.v1.html(html_exp, height=600, scrolling=True)
                    
                    # Feature contributions
                    lime_values = exp.as_list()
                    lime_df = pd.DataFrame(lime_values, columns=['Feature', 'Contribution'])
                    lime_df = lime_df.sort_values('Contribution', key=abs, ascending=True)
                    
                    fig_lime = px.bar(
                        lime_df,
                        x='Contribution',
                        y='Feature',
                        orientation='h',
                        title='LIME Feature Contributions',
                        color='Contribution',
                        color_continuous_scale='RdBu_r'
                    )
                    fig_lime.update_layout(height=400)
                    st.plotly_chart(fig_lime, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error generating LIME explanation: {str(e)}")
    
    # Model Performance Comparison
    st.markdown('<h2 class="section-header">📈 Model Performance</h2>', unsafe_allow_html=True)
    
    with st.expander("📊 Detailed Performance Metrics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Logistic Regression")
            lr_pred = lr_model.predict(X_test)
            st.text(classification_report(y_test, lr_pred, target_names=['No Churn', 'Churn']))
        
        with col2:
            st.subheader("Decision Tree")
            dt_pred = dt_model.predict(X_test)
            st.text(classification_report(y_test, dt_pred, target_names=['No Churn', 'Churn']))
    
    # Footer
    st.markdown("---")
    st.markdown("### 📝 How to Use This Dashboard:")
    st.markdown("""
    1. **Model Selection**: Choose between Logistic Regression and Decision Tree models
    2. **SHAP Analysis**: Click 'Generate SHAP Plots' to see global feature importance and summary plots
    3. **LIME Analysis**: Select an instance and click 'Generate LIME Explanation' to see local explanations
    4. **Sample Size**: Adjust the sample size in the sidebar for different analysis scopes
    """)
    
else:
    st.error("Please ensure the dataset file is available in the correct location.")
    st.info("Expected file path: Data/customer_churn_dataset_prepared.csv")