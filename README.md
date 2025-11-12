# Student Dropout Prediction System

A comprehensive machine learning system for predicting student dropout risk and generating retention scores. The system includes data preprocessing, feature engineering, model training, explainability, and an interactive dashboard.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training a Model](#training-a-model)
  - [Running the API](#running-the-api)
  - [Using the Dashboard](#using-the-dashboard)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Student Dropout Prediction System is designed to help educational institutions identify students at risk of dropping out and provide targeted interventions. The system uses machine learning models to analyze various student factors and generate:

- Dropout probability predictions
- Retention scores (0-100)
- Risk categorization (Low, Medium, High)
- Personalized intervention recommendations
- Model explanations using SHAP and LIME

## Features

- **Data Collection & Preprocessing**: Automated data collection and preprocessing pipeline
- **Feature Engineering**: Advanced feature engineering to improve model performance
- **Model Training**: Multiple ML models with automatic selection of the best performer
- **Retention Scoring**: Conversion of dropout probability to retention scores
- **Explainability**: SHAP and LIME explanations for model predictions
- **API Service**: RESTful API for making predictions
- **Interactive Dashboard**: Streamlit-based dashboard for visualization and predictions

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/student-dropout-prediction.git
   cd student-dropout-prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training a Model

To train a new model using sample data:

```bash
python -m src.models.model_trainer
```

This will:
1. Generate sample student data
2. Preprocess the data
3. Apply feature engineering
4. Train multiple ML models
5. Select the best model based on ROC AUC
6. Save the model and preprocessors

### Running the API

To start the API server:

```bash
python -m uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`. You can access the interactive API documentation at `http://localhost:8000/docs`.

### Using the Dashboard

To start the Streamlit dashboard:

```bash
streamlit run src/dashboard/app.py
```

The dashboard will open in your browser and provide:
- Model overview and performance metrics
- Interactive prediction interface
- Model insights and feature importance
- Data exploration tools
- Model training interface

## Project Structure

```
student-dropout-prediction/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI service
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py               # Streamlit dashboard
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py
│   │   └── generate_sample_data.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_trainer.py     # Model training and evaluation
│   └── utils/
│       ├── __init__.py
│       ├── explainability.py    # SHAP/LIME explanations
│       ├── feature_engineering.py
│       └── retention_score.py   # Retention score calculator
├── data/                        # Data directory
├── models/                      # Saved models
├── notebooks/                   # Jupyter notebooks
├── requirements.txt
└── README.md
```

## API Documentation

### Health Check

Check if the API is running:

```bash
GET /health
```

### Load Model

Load the trained model:

```bash
POST /load-model
```

### Get Model Information

Get information about the loaded model:

```bash
GET /model-info
```

### Make Predictions

Predict dropout probability for students:

```bash
POST /predict
Content-Type: application/json

{
  "students": [
    {
      "student_id": "STU_00001",
      "age": 18,
      "gender": "Male",
      "family_income": "Medium",
      "parent_education": "Secondary",
      "family_size": 4,
      "rural_urban": "Urban",
      "first_generation": 0,
      "previous_grades": 75.0,
      "attendance_rate": 85.0,
      "extracurricular_activities": 2,
      "study_hours_per_week": 15,
      "subject_preferences": "Science",
      "learning_disability": 0,
      "special_education_needs": 0,
      "scholarship_received": 1,
      "scholarship_amount": 5000.0,
      "financial_aid": 0,
      "work_study_program": 0,
      "part_time_job": 0,
      "family_financial_support": "Medium",
      "debt_amount": 0.0,
      "interview_score": 7.5,
      "motivation_level": "Medium",
      "career_goals_clarity": 7,
      "emotional_stability": 8,
      "peer_relationships": "Good",
      "teacher_relationships": "Good",
      "disciplinary_issues": 1
    }
  ]
}
```

### Get Predictions with Recommendations

Get dropout predictions and intervention recommendations:

```bash
POST /predict-with-recommendations
```

Same request format as `/predict`, but includes intervention recommendations in the response.

### Get Feature Importance

Get global feature importance:

```bash
GET /feature-importance?method=shap&top_n=10
```

Parameters:
- `method`: Explanation method (`shap` or `lime`)
- `top_n`: Number of top features to return

## Model Performance

The system trains and evaluates multiple machine learning models:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost
5. Decision Tree
6. K-Nearest Neighbors
7. Naive Bayes
8. Support Vector Machine (SVM)

Models are evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

The best model is selected based on the ROC AUC score, which measures the model's ability to distinguish between students who will drop out and those who will not.

### Expected Performance

Based on our testing with synthetic data, the best models typically achieve:
- ROC AUC: 0.85-0.92
- Accuracy: 0.80-0.88
- F1 Score: 0.78-0.85

Performance may vary with real-world data depending on data quality and feature relevance.

## Retention Score Calculation

The system converts dropout probability to a retention score out of 100 using the formula:

```
Retention Score = (1 - Dropout Probability) × 100
```

Students are then categorized into risk levels:
- **Low risk / Highly deserving**: Retention score ≥ 80
- **Medium risk / Needs monitoring**: 50 ≤ Retention score < 80
- **High dropout risk / Needs intervention**: Retention score < 50

## Intervention Recommendations

The system provides personalized intervention recommendations based on a student's risk category and individual characteristics:

### Low Risk Students
- Continue academic support
- Provide enrichment opportunities
- Consider for leadership roles
- Monthly check-ins

### Medium Risk Students
- Schedule regular counseling sessions
- Monitor academic progress closely
- Address specific risk factors (attendance, grades, financial, etc.)
- Bi-weekly check-ins

### High Risk Students
- Immediate intervention required
- Develop personalized success plan
- Assign dedicated counselor
- Address urgent risk factors
- Weekly check-ins

## Model Explainability

The system provides model explanations using two methods:

### SHAP (SHapley Additive exPlanations)
- Global feature importance
- Local prediction explanations
- Dependence plots
- Summary plots

### LIME (Local Interpretable Model-agnostic Explanations)
- Local prediction explanations
- Feature importance for individual predictions

These explanations help educators understand which factors are most influential in predicting dropout risk for individual students and across the population.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The development team at Maatram Foundation
- Educational institutions that provided insights and requirements
- Open source libraries and tools that made this project possible