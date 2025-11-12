import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetentionScoreCalculator:
    """
    A class for calculating retention/deservingness scores from dropout predictions.
    Converts dropout probability into a score out of 100 and categorizes students by risk level.
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.feature_names = None
        
    def load_model_and_preprocessors(self, model_path, preprocessor_path=None, feature_engineer_path=None):
        """
        Load the trained model and preprocessors.
        
        Args:
            model_path (str): Path to the saved model.
            preprocessor_path (str): Path to the saved preprocessor.
            feature_engineer_path (str): Path to the saved feature engineer.
        """
        logger.info("Loading model and preprocessors")
        
        # Load model
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        
        # Load preprocessor if provided
        if preprocessor_path:
            self.preprocessor = joblib.load(preprocessor_path)
        
        # Load feature engineer if provided
        if feature_engineer_path:
            self.feature_engineer = joblib.load(feature_engineer_path)
    
    def calculate_retention_score(self, dropout_probability):
        """
        Convert dropout probability to retention score out of 100.
        
        Args:
            dropout_probability (float or np.array): Dropout probability(ies).
            
        Returns:
            float or np.array: Retention score(s) out of 100.
        """
        # Convert dropout probability to retention score
        # Higher dropout probability = lower retention score
        retention_score = (1 - dropout_probability) * 100
        
        return retention_score
    
    def categorize_risk_level(self, retention_score):
        """
        Categorize students based on retention score.
        
        Args:
            retention_score (float or np.array): Retention score(s).
            
        Returns:
            str or np.array: Risk level category(ies).
        """
        if isinstance(retention_score, (list, np.ndarray)):
            risk_levels = []
            for score in retention_score:
                if score >= 80:
                    risk_levels.append("Low risk / Highly deserving")
                elif score >= 50:
                    risk_levels.append("Medium risk / Needs monitoring")
                else:
                    risk_levels.append("High dropout risk / Needs intervention")
            return np.array(risk_levels)
        else:
            if retention_score >= 80:
                return "Low risk / Highly deserving"
            elif retention_score >= 50:
                return "Medium risk / Needs monitoring"
            else:
                return "High dropout risk / Needs intervention"
    
    def predict_dropout_probability(self, student_data):
        """
        Predict dropout probability for student data.
        
        Args:
            student_data (pd.DataFrame): Student data for prediction.
            
        Returns:
            np.array: Dropout probabilities.
        """
        # Apply feature engineering if available
        if self.feature_engineer:
            student_data = self.feature_engineer.engineer_features(student_data.copy())
        
        # Apply preprocessing if available
        if self.preprocessor:
            # Drop student_id if present
            if 'student_id' in student_data.columns:
                student_ids = student_data['student_id']
                student_data = student_data.drop(columns=['student_id'])
            else:
                student_ids = None
            
            # Preprocess data
            processed_data = self.preprocessor.transform(student_data)
            
            # Ensure feature names match
            if hasattr(processed_data, 'columns'):
                processed_data = processed_data[self.feature_names]
            else:
                # If it's a numpy array, we assume the order is correct
                pass
        else:
            # If no preprocessor, use data as is
            processed_data = student_data[self.feature_names]
        
        # Predict dropout probability
        dropout_probabilities = self.model.predict_proba(processed_data)[:, 1]
        
        return dropout_probabilities
    
    def calculate_retention_scores(self, student_data):
        """
        Calculate retention scores and risk categories for student data.
        
        Args:
            student_data (pd.DataFrame): Student data for prediction.
            
        Returns:
            pd.DataFrame: DataFrame with retention scores and risk categories.
        """
        logger.info("Calculating retention scores")
        
        # Predict dropout probabilities
        dropout_probabilities = self.predict_dropout_probability(student_data)
        
        # Calculate retention scores
        retention_scores = self.calculate_retention_score(dropout_probabilities)
        
        # Categorize risk levels
        risk_categories = self.categorize_risk_level(retention_scores)
        
        # Create results dataframe
        results = pd.DataFrame({
            'student_id': student_data['student_id'] if 'student_id' in student_data.columns else range(len(student_data)),
            'dropout_probability': dropout_probabilities,
            'retention_score': retention_scores,
            'risk_category': risk_categories
        })
        
        return results
    
    def get_intervention_recommendations(self, student_data):
        """
        Generate intervention recommendations based on risk categories.
        
        Args:
            student_data (pd.DataFrame): Student data with retention scores.
            
        Returns:
            pd.DataFrame: DataFrame with intervention recommendations.
        """
        logger.info("Generating intervention recommendations")
        
        # Calculate retention scores if not already present
        if 'retention_score' not in student_data.columns:
            results = self.calculate_retention_scores(student_data)
        else:
            results = student_data.copy()
        
        # Define recommendations based on risk categories
        def get_recommendation(risk_category, student_data_row):
            if risk_category == "Low risk / Highly deserving":
                return {
                    'priority': 'Low',
                    'actions': [
                        'Continue academic support',
                        'Provide enrichment opportunities',
                        'Consider for leadership roles'
                    ],
                    'frequency': 'Monthly check-ins'
                }
            elif risk_category == "Medium risk / Needs monitoring":
                # Analyze key factors contributing to risk
                recommendations = {
                    'priority': 'Medium',
                    'actions': [
                        'Schedule regular counseling sessions',
                        'Monitor academic progress closely'
                    ],
                    'frequency': 'Bi-weekly check-ins'
                }
                
                # Add specific recommendations based on student data
                if 'attendance_rate' in student_data_row and student_data_row['attendance_rate'] < 75:
                    recommendations['actions'].append('Address attendance issues')
                
                if 'previous_grades' in student_data_row and student_data_row['previous_grades'] < 60:
                    recommendations['actions'].append('Provide academic tutoring')
                
                if 'family_income' in student_data_row and student_data_row['family_income'] == 'Low':
                    recommendations['actions'].append('Review financial aid options')
                
                return recommendations
            else:  # High dropout risk / Needs intervention
                recommendations = {
                    'priority': 'High',
                    'actions': [
                        'Immediate intervention required',
                        'Develop personalized success plan',
                        'Assign dedicated counselor'
                    ],
                    'frequency': 'Weekly check-ins'
                }
                
                # Add specific recommendations based on student data
                if 'attendance_rate' in student_data_row and student_data_row['attendance_rate'] < 60:
                    recommendations['actions'].append('Urgent: Address severe attendance issues')
                
                if 'previous_grades' in student_data_row and student_data_row['previous_grades'] < 50:
                    recommendations['actions'].append('Urgent: Provide intensive academic support')
                
                if 'family_income' in student_data_row and student_data_row['family_income'] == 'Low':
                    recommendations['actions'].append('Urgent: Secure financial assistance')
                
                if 'emotional_stability' in student_data_row and student_data_row['emotional_stability'] < 5:
                    recommendations['actions'].append('Urgent: Mental health support')
                
                return recommendations
        
        # Apply recommendations to each student
        recommendations_list = []
        for _, row in results.iterrows():
            rec = get_recommendation(row['risk_category'], row)
            recommendations_list.append(rec)
        
        # Add recommendations to results
        results['recommendations'] = recommendations_list
        
        return results
    
    def generate_summary_statistics(self, student_data):
        """
        Generate summary statistics for the student data.
        
        Args:
            student_data (pd.DataFrame): Student data with retention scores.
            
        Returns:
            dict: Summary statistics.
        """
        logger.info("Generating summary statistics")
        
        # Calculate retention scores if not already present
        if 'retention_score' not in student_data.columns:
            results = self.calculate_retention_scores(student_data)
        else:
            results = student_data.copy()
        
        # Calculate statistics
        stats = {
            'total_students': len(results),
            'average_retention_score': results['retention_score'].mean(),
            'median_retention_score': results['retention_score'].median(),
            'min_retention_score': results['retention_score'].min(),
            'max_retention_score': results['retention_score'].max(),
            'risk_distribution': results['risk_category'].value_counts().to_dict(),
            'risk_percentage': (results['risk_category'].value_counts(normalize=True) * 100).round(2).to_dict()
        }
        
        return stats
    
    def save_calculator(self, filepath):
        """
        Save the retention score calculator to a file.
        
        Args:
            filepath (str): Path to save the calculator.
        """
        logger.info(f"Saving retention score calculator to {filepath}")
        
        calculator_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_engineer': self.feature_engineer,
            'feature_names': self.feature_names
        }
        
        joblib.dump(calculator_data, filepath)
    
    def load_calculator(self, filepath):
        """
        Load a retention score calculator from a file.
        
        Args:
            filepath (str): Path to the saved calculator.
            
        Returns:
            dict: Loaded calculator data.
        """
        logger.info(f"Loading retention score calculator from {filepath}")
        
        calculator_data = joblib.load(filepath)
        
        self.model = calculator_data['model']
        self.preprocessor = calculator_data.get('preprocessor')
        self.feature_engineer = calculator_data.get('feature_engineer')
        self.feature_names = calculator_data['feature_names']
        
        return calculator_data


def main():
    """
    Example usage of the RetentionScoreCalculator class.
    """
    # Import required modules
    from src.data.generate_sample_data import generate_sample_data
    from src.data.data_preprocessing import DataPreprocessor
    from src.utils.feature_engineering import FeatureEngineer
    from src.models.model_trainer import ModelTrainer
    
    # Generate sample data
    logger.info("Generating sample data")
    data = generate_sample_data(num_students=100, output_dir='data')
    df = data['combined']
    
    # Preprocess data
    logger.info("Preprocessing data")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_data(df)
    
    # Feature engineering
    logger.info("Applying feature engineering")
    feature_engineer = FeatureEngineer()
    df_engineered = feature_engineer.engineer_features(df.copy())
    
    # Re-preprocess engineered data
    X_engineered, y = preprocessor.preprocess_data(df_engineered)
    
    # Train model
    logger.info("Training model")
    trainer = ModelTrainer()
    trainer.initialize_models()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X_engineered, y)
    
    # Train models
    trainer.train_models(X_train, X_val, y_train, y_val)
    
    # Save model and preprocessors
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/best_model.joblib')
    preprocessor.save_preprocessor('models/preprocessor.joblib')
    feature_engineer.save_pipeline('models/feature_engineering_pipeline.joblib')
    
    # Initialize retention score calculator
    calculator = RetentionScoreCalculator()
    calculator.load_model_and_preprocessors(
        model_path='models/best_model.joblib',
        preprocessor_path='models/preprocessor.joblib',
        feature_engineer_path='models/feature_engineering_pipeline.joblib'
    )
    
    # Calculate retention scores for test data
    test_df = df_engineered.iloc[:10].copy()  # Use first 10 students for testing
    results = calculator.calculate_retention_scores(test_df)
    
    # Generate intervention recommendations
    recommendations = calculator.get_intervention_recommendations(test_df)
    
    # Generate summary statistics
    stats = calculator.generate_summary_statistics(test_df)
    
    # Save calculator
    calculator.save_calculator('models/retention_calculator.joblib')
    
    # Display results
    print("\nRetention Scores:")
    print(results[['student_id', 'retention_score', 'risk_category']])
    
    print("\nSummary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    logger.info("Retention score calculation completed successfully")


if __name__ == "__main__":
    main()