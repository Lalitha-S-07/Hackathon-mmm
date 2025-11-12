import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import os
import sys
from datetime import datetime

# Add src to path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from src.data.generate_sample_data import generate_sample_data
from src.data.data_preprocessing import DataPreprocessor
from src.utils.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.utils.retention_score import RetentionScoreCalculator
from src.utils.explainability import ModelExplainer

# Configure page
st.set_page_config(
    page_title="Student Dropout Prediction Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define API URL
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #42A5F5;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F5F7FA;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 0.1rem 0.2rem rgba(0,0,0,0.1);
    }
    .risk-low {
        background-color: #C8E6C9;
        color: #2E7D32;
    }
    .risk-medium {
        background-color: #FFF9C4;
        color: #F57F17;
    }
    .risk-high {
        background-color: #FFCDD2;
        color: #C62828;
    }
    .stDataFrame {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API returned status code {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

def load_model():
    """Load the model via API"""
    try:
        response = requests.post(f"{API_URL}/load-model")
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API returned status code {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_URL}/model-info")
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API returned status code {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

def predict_dropout(student_data):
    """Predict dropout probability for a student"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"students": [student_data]}
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API returned status code {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

def predict_with_recommendations(student_data):
    """Predict dropout probability and get recommendations"""
    try:
        response = requests.post(
            f"{API_URL}/predict-with-recommendations",
            json={"students": [student_data]}
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API returned status code {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

def get_feature_importance(method="shap", top_n=10):
    """Get feature importance"""
    try:
        response = requests.get(
            f"{API_URL}/feature-importance",
            params={"method": method, "top_n": top_n}
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API returned status code {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

def generate_sample_data(num_students=100):
    """Generate sample data for demonstration"""
    data = generate_sample_data(num_students=num_students, output_dir="data")
    return data['combined']

def train_model_locally():
    """Train a model locally (for demonstration purposes)"""
    # Generate sample data
    data = generate_sample_data(num_students=1000, output_dir="data")
    df = data['combined']
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_data(df)
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    df_engineered = feature_engineer.engineer_features(df.copy())
    
    # Re-preprocess engineered data
    X_engineered, y = preprocessor.preprocess_data(df_engineered)
    
    # Train model
    trainer = ModelTrainer()
    trainer.initialize_models()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X_engineered, y)
    
    # Train models
    trainer.train_models(X_train, X_val, y_train, y_val)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    trainer.save_model("models/best_model.joblib")
    preprocessor.save_preprocessor("models/preprocessor.joblib")
    feature_engineer.save_pipeline("models/feature_engineering_pipeline.joblib")
    
    return True, "Model trained successfully"

# Main app
def main():
    # Main header
    st.markdown("<h1 class='main-header'>Student Dropout Prediction Dashboard</h1>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Dashboard Overview", 
        "Make Predictions", 
        "Model Insights",
        "Data Exploration",
        "Model Training"
    ])
    
    # Check API health
    api_healthy, health_info = check_api_health()
    
    if not api_healthy:
        st.error(f"API is not running: {health_info.get('error', 'Unknown error')}")
        st.info("Please start the API server by running: `python -m uvicorn src.api.main:app --reload`")
    
    # Dashboard Overview page
    if page == "Dashboard Overview":
        st.markdown("<h2 class='sub-header'>Dashboard Overview</h2>", unsafe_allow_html=True)
        
        # Display API status
        col1, col2 = st.columns(2)
        with col1:
            if api_healthy:
                st.success("API Status: ✅ Running")
            else:
                st.error("API Status: ❌ Not Running")
        
        with col2:
            if api_healthy:
                st.write(f"API Version: {health_info.get('api_version', 'Unknown')}")
                st.write(f"Timestamp: {health_info.get('timestamp', 'Unknown')}")
            else:
                st.write("API Version: Unknown")
                st.write("Timestamp: Unknown")
        
        # Model information
        st.markdown("<h3 class='sub-header'>Model Information</h3>", unsafe_allow_html=True)
        
        if api_healthy:
            model_loaded = health_info.get('model_loaded', False)
            if model_loaded:
                # Get model info
                model_info_success, model_info = get_model_info()
                if model_info_success:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model Name", model_info.get('model_name', 'Unknown'))
                    with col2:
                        st.metric("Feature Count", model_info.get('feature_count', 0))
                    with col3:
                        st.metric("Last Trained", model_info.get('last_trained', 'Unknown'))
                    
                    # Display performance metrics
                    st.markdown("<h4>Performance Metrics</h4>", unsafe_allow_html=True)
                    metrics = model_info.get('performance_metrics', {})
                    if metrics:
                        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                        st.dataframe(metrics_df)
                    else:
                        st.info("No performance metrics available")
                else:
                    st.error(f"Failed to get model info: {model_info.get('error', 'Unknown error')}")
            else:
                st.warning("Model not loaded. Please load the model using the button below.")
                if st.button("Load Model"):
                    load_success, load_result = load_model()
                    if load_success:
                        st.success("Model loaded successfully!")
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to load model: {load_result.get('error', 'Unknown error')}")
        else:
            st.warning("API is not running. Cannot display model information.")
        
        # Feature importance
        st.markdown("<h3 class='sub-header'>Feature Importance</h3>", unsafe_allow_html=True)
        
        if api_healthy:
            # Get feature importance
            importance_success, importance_data = get_feature_importance()
            if importance_success:
                importance_df = pd.DataFrame(importance_data.get('feature_importance', []))
                if not importance_df.empty:
                    # Create bar chart
                    fig = px.bar(
                        importance_df, 
                        x='importance', 
                        y='feature', 
                        orientation='h',
                        title=f"Top Features ({importance_data.get('method', 'shap').upper()})",
                        color='importance',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No feature importance data available")
            else:
                st.error(f"Failed to get feature importance: {importance_data.get('error', 'Unknown error')}")
        else:
            st.warning("API is not running. Cannot display feature importance.")
    
    # Make Predictions page
    elif page == "Make Predictions":
        st.markdown("<h2 class='sub-header'>Make Predictions</h2>", unsafe_allow_html=True)
        
        if not api_healthy:
            st.error("API is not running. Please start the API server to make predictions.")
            st.stop()
        
        # Check if model is loaded
        model_loaded = health_info.get('model_loaded', False)
        if not model_loaded:
            st.warning("Model not loaded. Please load the model first.")
            if st.button("Load Model"):
                load_success, load_result = load_model()
                if load_success:
                    st.success("Model loaded successfully!")
                    st.experimental_rerun()
                else:
                    st.error(f"Failed to load model: {load_result.get('error', 'Unknown error')}")
            st.stop()
        
        # Input form
        st.markdown("<h3>Student Information</h3>", unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            # Demographic information
            st.markdown("<h4>Demographic Information</h4>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                student_id = st.text_input("Student ID", value="STU_00001")
                age = st.slider("Age", min_value=16, max_value=25, value=18)
                gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
                family_income = st.selectbox("Family Income", options=["Low", "Medium", "High"])
                parent_education = st.selectbox("Parent Education", options=["Primary", "Secondary", "Graduate", "Postgraduate"])
            
            with col2:
                family_size = st.slider("Family Size", min_value=1, max_value=10, value=4)
                rural_urban = st.selectbox("Location", options=["Rural", "Urban"])
                first_generation = st.checkbox("First Generation Student")
            
            # Academic information
            st.markdown("<h4>Academic Information</h4>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                previous_grades = st.slider("Previous Grades (%)", min_value=0, max_value=100, value=75)
                attendance_rate = st.slider("Attendance Rate (%)", min_value=0, max_value=100, value=85)
                extracurricular_activities = st.slider("Extracurricular Activities", min_value=0, max_value=10, value=2)
                study_hours_per_week = st.slider("Study Hours per Week", min_value=0, max_value=50, value=15)
            
            with col2:
                subject_preferences = st.selectbox("Subject Preferences", options=["Science", "Arts", "Commerce", "Vocational"])
                learning_disability = st.checkbox("Learning Disability")
                special_education_needs = st.checkbox("Special Education Needs")
            
            # Financial information
            st.markdown("<h4>Financial Information</h4>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                scholarship_received = st.checkbox("Scholarship Received")
                if scholarship_received:
                    scholarship_amount = st.number_input("Scholarship Amount", min_value=0, value=5000)
                else:
                    scholarship_amount = 0
                
                financial_aid = st.checkbox("Financial Aid Received")
            
            with col2:
                work_study_program = st.checkbox("Work-Study Program")
                part_time_job = st.checkbox("Part-time Job")
                family_financial_support = st.selectbox("Family Financial Support", options=["Low", "Medium", "High"])
                debt_amount = st.number_input("Debt Amount", min_value=0, value=0)
            
            # Interview and behavioral information
            st.markdown("<h4>Interview and Behavioral Information</h4>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                interview_score = st.slider("Interview Score", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
                motivation_level = st.selectbox("Motivation Level", options=["Low", "Medium", "High"])
                career_goals_clarity = st.slider("Career Goals Clarity", min_value=1, max_value=10, value=7)
                emotional_stability = st.slider("Emotional Stability", min_value=1, max_value=10, value=8)
            
            with col2:
                peer_relationships = st.selectbox("Peer Relationships", options=["Poor", "Average", "Good"])
                teacher_relationships = st.selectbox("Teacher Relationships", options=["Poor", "Average", "Good"])
                disciplinary_issues = st.slider("Disciplinary Issues", min_value=0, max_value=10, value=1)
            
            # Submit button
            submit_button = st.form_submit_button("Predict Dropout Risk")
        
        # Make prediction
        if submit_button:
            # Prepare student data
            student_data = {
                "student_id": student_id,
                "age": age,
                "gender": gender,
                "family_income": family_income,
                "parent_education": parent_education,
                "family_size": family_size,
                "rural_urban": rural_urban,
                "first_generation": 1 if first_generation else 0,
                "previous_grades": previous_grades,
                "attendance_rate": attendance_rate,
                "extracurricular_activities": extracurricular_activities,
                "study_hours_per_week": study_hours_per_week,
                "subject_preferences": subject_preferences,
                "learning_disability": 1 if learning_disability else 0,
                "special_education_needs": 1 if special_education_needs else 0,
                "scholarship_received": 1 if scholarship_received else 0,
                "scholarship_amount": float(scholarship_amount),
                "financial_aid": 1 if financial_aid else 0,
                "work_study_program": 1 if work_study_program else 0,
                "part_time_job": 1 if part_time_job else 0,
                "family_financial_support": family_financial_support,
                "debt_amount": float(debt_amount),
                "interview_score": interview_score,
                "motivation_level": motivation_level,
                "career_goals_clarity": career_goals_clarity,
                "emotional_stability": emotional_stability,
                "peer_relationships": peer_relationships,
                "teacher_relationships": teacher_relationships,
                "disciplinary_issues": disciplinary_issues
            }
            
            # Get prediction with recommendations
            success, result = predict_with_recommendations(student_data)
            
            if success:
                prediction = result[0]
                
                # Display prediction results
                st.markdown("<h3>Prediction Results</h3>", unsafe_allow_html=True)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    dropout_prob = prediction.get('dropout_probability', 0) * 100
                    st.metric("Dropout Probability", f"{dropout_prob:.1f}%")
                
                with col2:
                    retention_score = prediction.get('retention_score', 0)
                    st.metric("Retention Score", f"{retention_score:.1f}/100")
                
                with col3:
                    risk_category = prediction.get('risk_category', 'Unknown')
                    if "Low risk" in risk_category:
                        st.markdown(f"<div class='metric-card risk-low'>Risk Category: {risk_category}</div>", unsafe_allow_html=True)
                    elif "Medium risk" in risk_category:
                        st.markdown(f"<div class='metric-card risk-medium'>Risk Category: {risk_category}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='metric-card risk-high'>Risk Category: {risk_category}</div>", unsafe_allow_html=True)
                
                # Display recommendations
                st.markdown("<h3>Intervention Recommendations</h3>", unsafe_allow_html=True)
                
                recommendations = prediction.get('recommendations', {})
                if recommendations:
                    priority = recommendations.get('priority', 'Unknown')
                    st.write(f"**Priority Level:** {priority}")
                    
                    actions = recommendations.get('actions', [])
                    if actions:
                        st.write("**Recommended Actions:**")
                        for action in actions:
                            st.write(f"- {action}")
                    
                    frequency = recommendations.get('frequency', 'Unknown')
                    st.write(f"**Check-in Frequency:** {frequency}")
                else:
                    st.info("No recommendations available")
                
                # Display prediction confidence
                st.markdown("<h3>Prediction Details</h3>", unsafe_allow_html=True)
                
                # Create gauge chart for retention score
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = retention_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Retention Score"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"Failed to make prediction: {result.get('error', 'Unknown error')}")
    
    # Model Insights page
    elif page == "Model Insights":
        st.markdown("<h2 class='sub-header'>Model Insights</h2>", unsafe_allow_html=True)
        
        if not api_healthy:
            st.error("API is not running. Please start the API server to view model insights.")
            st.stop()
        
        # Check if model is loaded
        model_loaded = health_info.get('model_loaded', False)
        if not model_loaded:
            st.warning("Model not loaded. Please load the model first.")
            if st.button("Load Model"):
                load_success, load_result = load_model()
                if load_success:
                    st.success("Model loaded successfully!")
                    st.experimental_rerun()
                else:
                    st.error(f"Failed to load model: {load_result.get('error', 'Unknown error')}")
            st.stop()
        
        # Feature importance
        st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
        
        # Method selection
        method = st.selectbox("Explanation Method", options=["shap", "lime"])
        top_n = st.slider("Number of Top Features", min_value=5, max_value=20, value=10)
        
        # Get feature importance
        importance_success, importance_data = get_feature_importance(method=method, top_n=top_n)
        if importance_success:
            importance_df = pd.DataFrame(importance_data.get('feature_importance', []))
            if not importance_df.empty:
                # Create bar chart
                fig = px.bar(
                    importance_df, 
                    x='importance', 
                    y='feature', 
                    orientation='h',
                    title=f"Top Features ({method.upper()})",
                    color='importance',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Display feature importance table
                st.write("Feature Importance Table:")
                st.dataframe(importance_df)
            else:
                st.info("No feature importance data available")
        else:
            st.error(f"Failed to get feature importance: {importance_data.get('error', 'Unknown error')}")
        
        # Model performance
        st.markdown("<h3>Model Performance</h3>", unsafe_allow_html=True)
        
        # Get model info
        model_info_success, model_info = get_model_info()
        if model_info_success:
            metrics = model_info.get('performance_metrics', {})
            if metrics:
                # Create metrics dataframe
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                
                # Create bar chart for metrics
                fig = px.bar(
                    metrics_df, 
                    x='Metric', 
                    y='Value',
                    title="Model Performance Metrics",
                    color='Value',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display metrics table
                st.write("Performance Metrics Table:")
                st.dataframe(metrics_df)
            else:
                st.info("No performance metrics available")
        else:
            st.error(f"Failed to get model info: {model_info.get('error', 'Unknown error')}")
    
    # Data Exploration page
    elif page == "Data Exploration":
        st.markdown("<h2 class='sub-header'>Data Exploration</h2>", unsafe_allow_html=True)
        
        # Generate or load data
        data_option = st.radio("Select data source:", options=["Generate Sample Data", "Upload CSV File"])
        
        if data_option == "Generate Sample Data":
            num_students = st.slider("Number of students to generate:", min_value=50, max_value=1000, value=200)
            if st.button("Generate Data"):
                with st.spinner("Generating sample data..."):
                    df = generate_sample_data(num_students=num_students)
                    st.success(f"Generated data for {num_students} students!")
                    st.session_state['data'] = df
        else:
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded data with {len(df)} rows!")
                st.session_state['data'] = df
        
        # Display data if available
        if 'data' in st.session_state:
            df = st.session_state['data']
            
            # Display data preview
            st.markdown("<h3>Data Preview</h3>", unsafe_allow_html=True)
            st.dataframe(df.head())
            
            # Display data info
            st.markdown("<h3>Data Information</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Number of students: {len(df)}")
                st.write(f"Number of features: {len(df.columns)}")
            
            with col2:
                if 'dropout_status' in df.columns:
                    dropout_rate = df['dropout_status'].mean() * 100
                    st.write(f"Dropout rate: {dropout_rate:.1f}%")
            
            # Display feature distributions
            st.markdown("<h3>Feature Distributions</h3>", unsafe_allow_html=True)
            
            # Select feature to display
            numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_features:
                selected_feature = st.selectbox("Select a feature to display:", options=numeric_features)
                
                # Create histogram
                fig = px.histogram(
                    df, 
                    x=selected_feature,
                    title=f"Distribution of {selected_feature}",
                    color_discrete_sequence=['#1E88E5']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display correlation matrix
            st.markdown("<h3>Correlation Matrix</h3>", unsafe_allow_html=True)
            
            if len(numeric_features) > 1:
                # Calculate correlation matrix
                corr_matrix = df[numeric_features].corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough numeric features to display correlation matrix")
    
    # Model Training page
    elif page == "Model Training":
        st.markdown("<h2 class='sub-header'>Model Training</h2>", unsafe_allow_html=True)
        
        st.info("""
        This section allows you to train a new model using sample data. 
        The trained model will be saved and can be used for predictions.
        """)
        
        if st.button("Train New Model"):
            with st.spinner("Training model... This may take a few minutes."):
                success, message = train_model_locally()
                if success:
                    st.success(message)
                    st.info("Model trained and saved successfully! You can now use it for predictions.")
                else:
                    st.error(f"Failed to train model: {message}")
        
        st.markdown("<h3>Model Training Details</h3>", unsafe_allow_html=True)
        
        st.write("""
        The model training process involves the following steps:
        
        1. **Data Generation**: Generate sample student data with various features.
        2. **Data Preprocessing**: Clean and preprocess the data for modeling.
        3. **Feature Engineering**: Create new features to improve model performance.
        4. **Model Training**: Train multiple machine learning models and select the best one.
        5. **Model Evaluation**: Evaluate the model using various metrics.
        6. **Model Saving**: Save the trained model for future use.
        """)
        
        st.write("""
        The following models are trained and compared:
        
        - Logistic Regression
        - Random Forest
        - Gradient Boosting
        - XGBoost
        - Decision Tree
        - K-Nearest Neighbors
        - Naive Bayes
        - Support Vector Machine (SVM)
        """)
        
        st.write("""
        The best model is selected based on the ROC AUC score, which measures the model's ability to distinguish between students who will drop out and those who will not.
        """)

if __name__ == "__main__":
    main()