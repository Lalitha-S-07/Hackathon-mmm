from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.retention_score import RetentionScoreCalculator
from src.utils.explainability import ModelExplainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Student Dropout Prediction API",
    description="API for predicting student dropout likelihood and generating retention scores",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessors
model_loaded = False
retention_calculator = None
model_explainer = None
model_info = {}

# Pydantic models for request/response
class StudentData(BaseModel):
    student_id: str = Field(..., description="Unique identifier for the student")
    age: int = Field(..., ge=16, le=25, description="Age of the student")
    gender: str = Field(..., description="Gender of the student")
    family_income: str = Field(..., description="Family income category")
    parent_education: str = Field(..., description="Highest education level of parents")
    family_size: int = Field(..., ge=1, le=10, description="Number of family members")
    rural_urban: str = Field(..., description="Rural or urban location")
    first_generation: int = Field(..., ge=0, le=1, description="First generation student indicator")
    previous_grades: float = Field(..., ge=0, le=100, description="Previous academic grades")
    attendance_rate: float = Field(..., ge=0, le=100, description="Attendance rate percentage")
    extracurricular_activities: int = Field(..., ge=0, le=10, description="Number of extracurricular activities")
    study_hours_per_week: int = Field(..., ge=0, le=50, description="Hours spent studying per week")
    subject_preferences: str = Field(..., description="Preferred subject area")
    learning_disability: int = Field(..., ge=0, le=1, description="Learning disability indicator")
    special_education_needs: int = Field(..., ge=0, le=1, description="Special education needs indicator")
    scholarship_received: int = Field(..., ge=0, le=1, description="Scholarship received indicator")
    scholarship_amount: float = Field(..., ge=0, description="Amount of scholarship received")
    financial_aid: int = Field(..., ge=0, le=1, description="Financial aid received indicator")
    work_study_program: int = Field(..., ge=0, le=1, description="Participation in work-study program")
    part_time_job: int = Field(..., ge=0, le=1, description="Has a part-time job")
    family_financial_support: str = Field(..., description="Level of family financial support")
    debt_amount: float = Field(..., ge=0, description="Amount of debt")
    interview_score: float = Field(..., ge=0, le=10, description="Interview score")
    motivation_level: str = Field(..., description="Motivation level")
    career_goals_clarity: int = Field(..., ge=1, le=10, description="Clarity of career goals")
    emotional_stability: int = Field(..., ge=1, le=10, description="Emotional stability score")
    peer_relationships: str = Field(..., description="Quality of peer relationships")
    teacher_relationships: str = Field(..., description="Quality of teacher relationships")
    disciplinary_issues: int = Field(..., ge=0, le=10, description="Number of disciplinary issues")

class PredictionRequest(BaseModel):
    students: List[StudentData] = Field(..., description="List of student data for prediction")

class PredictionResponse(BaseModel):
    student_id: str
    dropout_probability: float
    retention_score: float
    risk_category: str
    prediction_confidence: Optional[float] = None

class ExplanationRequest(BaseModel):
    student_id: str
    method: str = Field("shap", description="Explanation method (shap or lime)")
    num_features: int = Field(10, ge=1, le=20, description="Number of features to include in explanation")

class ExplanationResponse(BaseModel):
    student_id: str
    method: str
    top_features: List[Dict[str, Any]]
    expected_value: Optional[float] = None
    predicted_probability: Optional[float] = None

class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    feature_count: int
    last_trained: str
    performance_metrics: Dict[str, float]

class RetentionStatsResponse(BaseModel):
    total_students: int
    average_retention_score: float
    risk_distribution: Dict[str, int]
    risk_percentage: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    api_version: str

# Helper functions
def load_model():
    """Load the trained model and preprocessors"""
    global model_loaded, retention_calculator, model_explainer, model_info
    
    try:
        # Define model paths
        model_path = "models/best_model.joblib"
        preprocessor_path = "models/preprocessor.joblib"
        feature_engineer_path = "models/feature_engineering_pipeline.joblib"
        
        # Check if model files exist
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Initialize retention calculator
        retention_calculator = RetentionScoreCalculator()
        retention_calculator.load_model_and_preprocessors(
            model_path=model_path,
            preprocessor_path=preprocessor_path if os.path.exists(preprocessor_path) else None,
            feature_engineer_path=feature_engineer_path if os.path.exists(feature_engineer_path) else None
        )
        
        # Initialize model explainer
        model_explainer = ModelExplainer()
        
        # Load model info
        model_data = joblib.load(model_path)
        model_info = {
            "model_name": model_data.get("model_name", "Unknown"),
            "model_version": "1.0.0",
            "feature_count": len(model_data.get("feature_names", [])),
            "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "performance_metrics": model_data.get("model_scores", {})
        }
        
        model_loaded = True
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def prepare_student_data(student_data_list):
    """Convert student data list to pandas DataFrame"""
    data = []
    for student in student_data_list:
        data.append(student.dict())
    
    return pd.DataFrame(data)

# API endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_loaded=model_loaded,
        api_version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_loaded=model_loaded,
        api_version="1.0.0"
    )

@app.post("/load-model")
async def load_model_endpoint():
    """Endpoint to load the model"""
    success = load_model()
    if success:
        return {"status": "success", "message": "Model loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    if not model_loaded:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    return ModelInfoResponse(**model_info)

@app.post("/predict", response_model=List[PredictionResponse])
async def predict_dropout(request: PredictionRequest):
    """Predict dropout probability for students"""
    if not model_loaded:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    try:
        # Prepare student data
        student_df = prepare_student_data(request.students)
        
        # Calculate retention scores
        results = retention_calculator.calculate_retention_scores(student_df)
        
        # Convert to response format
        response = []
        for _, row in results.iterrows():
            response.append(PredictionResponse(
                student_id=row['student_id'],
                dropout_probability=row['dropout_probability'],
                retention_score=row['retention_score'],
                risk_category=row['risk_category']
            ))
        
        return response
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-with-recommendations")
async def predict_with_recommendations(request: PredictionRequest):
    """Predict dropout probability and get intervention recommendations"""
    if not model_loaded:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    try:
        # Prepare student data
        student_df = prepare_student_data(request.students)
        
        # Calculate retention scores
        results = retention_calculator.calculate_retention_scores(student_df)
        
        # Get intervention recommendations
        recommendations = retention_calculator.get_intervention_recommendations(student_df)
        
        # Combine results
        response = []
        for _, row in results.iterrows():
            student_id = row['student_id']
            student_recommendations = recommendations[recommendations['student_id'] == student_id]['recommendations'].values[0]
            
            response.append({
                "student_id": student_id,
                "dropout_probability": float(row['dropout_probability']),
                "retention_score": float(row['retention_score']),
                "risk_category": row['risk_category'],
                "recommendations": student_recommendations
            })
        
        return response
        
    except Exception as e:
        logger.error(f"Error in prediction with recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(request: ExplanationRequest):
    """Get explanation for a specific student's prediction"""
    if not model_loaded:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    try:
        # Get student data (in a real app, you would fetch this from a database)
        # For now, we'll use a placeholder
        student_data = {
            "student_id": request.student_id,
            # Add other required fields with default values
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
        
        # Convert to DataFrame
        student_df = pd.DataFrame([student_data])
        
        # Get dropout probability
        dropout_prob = retention_calculator.predict_dropout_probability(student_df)[0]
        
        # Get explanation
        # Note: This is a simplified version. In a real app, you would need to properly
        # set up the explainer with training data.
        explanation = {
            "student_id": request.student_id,
            "method": request.method,
            "top_features": [
                {"feature": "attendance_rate", "importance": 0.25, "direction": "lower attendance increases dropout risk"},
                {"feature": "previous_grades", "importance": 0.20, "direction": "lower grades increase dropout risk"},
                {"feature": "family_income", "importance": 0.15, "direction": "lower income increases dropout risk"},
                {"feature": "motivation_level", "importance": 0.10, "direction": "lower motivation increases dropout risk"},
                {"feature": "emotional_stability", "importance": 0.08, "direction": "lower stability increases dropout risk"}
            ],
            "predicted_probability": float(dropout_prob)
        }
        
        return ExplanationResponse(**explanation)
        
    except Exception as e:
        logger.error(f"Error in explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

@app.post("/batch-stats", response_model=RetentionStatsResponse)
async def get_batch_stats(request: PredictionRequest):
    """Get summary statistics for a batch of students"""
    if not model_loaded:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    try:
        # Prepare student data
        student_df = prepare_student_data(request.students)
        
        # Calculate retention scores
        results = retention_calculator.calculate_retention_scores(student_df)
        
        # Generate summary statistics
        stats = retention_calculator.generate_summary_statistics(student_df)
        
        return RetentionStatsResponse(
            total_students=stats["total_students"],
            average_retention_score=stats["average_retention_score"],
            risk_distribution=stats["risk_distribution"],
            risk_percentage=stats["risk_percentage"]
        )
        
    except Exception as e:
        logger.error(f"Error in batch stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch stats error: {str(e)}")

@app.get("/feature-importance")
async def get_feature_importance(method: str = "shap", top_n: int = 10):
    """Get global feature importance"""
    if not model_loaded:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    try:
        # Note: In a real app, you would need to properly set up the explainer with training data
        # For now, we'll return a placeholder response
        feature_importance = [
            {"feature": "attendance_rate", "importance": 0.25},
            {"feature": "previous_grades", "importance": 0.20},
            {"feature": "family_income", "importance": 0.15},
            {"feature": "motivation_level", "importance": 0.10},
            {"feature": "emotional_stability", "importance": 0.08},
            {"feature": "study_hours_per_week", "importance": 0.07},
            {"feature": "family_size", "importance": 0.05},
            {"feature": "scholarship_received", "importance": 0.04},
            {"feature": "disciplinary_issues", "importance": 0.03},
            {"feature": "peer_relationships", "importance": 0.03}
        ]
        
        return {"method": method, "feature_importance": feature_importance[:top_n]}
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature importance error: {str(e)}")

# Background task to load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up API server")
    load_model()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)