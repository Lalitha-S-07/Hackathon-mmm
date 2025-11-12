import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    A class for feature engineering on student data.
    Creates meaningful derived features to improve model performance.
    """
    
    def __init__(self):
        self.feature_pipeline = None
        
    def create_academic_features(self, df):
        """
        Create academic-related features.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with new academic features.
        """
        logger.info("Creating academic features")
        
        # Academic performance categories
        df['academic_performance_category'] = pd.cut(
            df['previous_grades'], 
            bins=[0, 60, 75, 90, 100], 
            labels=['Poor', 'Average', 'Good', 'Excellent']
        )
        
        # Attendance categories
        df['attendance_category'] = pd.cut(
            df['attendance_rate'], 
            bins=[0, 60, 75, 90, 100], 
            labels=['Poor', 'Average', 'Good', 'Excellent']
        )
        
        # Study efficiency (grades per study hour)
        df['study_efficiency'] = df['previous_grades'] / df['study_hours_per_week'].replace(0, 1)
        
        # Academic engagement score (combination of attendance and extracurricular activities)
        df['academic_engagement'] = (
            (df['attendance_rate'] / 100) * 0.7 + 
            (df['extracurricular_activities'] / 5) * 0.3
        )
        
        return df
    
    def create_financial_features(self, df):
        """
        Create financial-related features.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with new financial features.
        """
        logger.info("Creating financial features")
        
        # Total financial support
        df['total_financial_support'] = df['scholarship_amount'] + (
            df['financial_aid'] * 5000  # Assuming average financial aid amount
        )
        
        # Financial burden (debt relative to family income)
        income_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        df['family_income_numeric'] = df['family_income'].map(income_mapping)
        df['financial_burden'] = df['debt_amount'] / (df['family_income_numeric'] * 10000)
        
        # Financial stability indicator
        df['financial_stability'] = (
            (df['scholarship_received'] * 0.4) + 
            (df['financial_aid'] * 0.3) + 
            (1 - df['work_study_program'] * 0.2) + 
            (1 - df['part_time_job'] * 0.1)
        )
        
        # Financial need level
        df['financial_need_level'] = pd.cut(
            df['total_financial_support'], 
            bins=[-1, 0, 5000, 10000, 20000], 
            labels=['No Support', 'Low', 'Medium', 'High']
        )
        
        return df
    
    def create_personal_features(self, df):
        """
        Create personal and behavioral features.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with new personal features.
        """
        logger.info("Creating personal features")
        
        # Motivation score (numeric)
        motivation_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        df['motivation_score'] = df['motivation_level'].map(motivation_mapping)
        
        # Relationship scores (numeric)
        peer_mapping = {'Poor': 1, 'Average': 2, 'Good': 3}
        teacher_mapping = {'Poor': 1, 'Average': 2, 'Good': 3}
        df['peer_relationship_score'] = df['peer_relationships'].map(peer_mapping)
        df['teacher_relationship_score'] = df['teacher_relationships'].map(teacher_mapping)
        
        # Overall social score
        df['social_score'] = (
            df['peer_relationship_score'] * 0.5 + 
            df['teacher_relationship_score'] * 0.5
        )
        
        # Personal stability score
        df['personal_stability'] = (
            (df['emotional_stability'] / 10) * 0.4 + 
            (df['motivation_score'] / 3) * 0.3 + 
            (df['career_goals_clarity'] / 10) * 0.3
        )
        
        # Risk factors count
        df['risk_factors_count'] = (
            (df['learning_disability'] == 1).astype(int) +
            (df['special_education_needs'] == 1).astype(int) +
            (df['disciplinary_issues'] > 3).astype(int) +
            (df['emotional_stability'] < 5).astype(int) +
            (df['motivation_level'] == 'Low').astype(int)
        )
        
        return df
    
    def create_demographic_features(self, df):
        """
        Create demographic-related features.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with new demographic features.
        """
        logger.info("Creating demographic features")
        
        # Parent education level (numeric)
        education_mapping = {'Primary': 1, 'Secondary': 2, 'Graduate': 3, 'Postgraduate': 4}
        df['parent_education_level'] = df['parent_education'].map(education_mapping)
        
        # Family responsibility index (inverse of family size, adjusted for first generation)
        df['family_responsibility'] = (
            (1 / df['family_size']) * (1 + df['first_generation'] * 0.5)
        )
        
        # Socioeconomic status (combination of family income and parent education)
        income_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        df['family_income_numeric'] = df['family_income'].map(income_mapping)
        df['socioeconomic_status'] = (
            df['family_income_numeric'] * 0.6 + 
            df['parent_education_level'] * 0.4
        )
        
        # Location disadvantage (rural and low income)
        df['location_disadvantage'] = (
            (df['rural_urban'] == 'Rural').astype(int) * 
            (df['family_income'] == 'Low').astype(int)
        )
        
        return df
    
    def create_interaction_features(self, df):
        """
        Create interaction features between different domains.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with new interaction features.
        """
        logger.info("Creating interaction features")
        
        # Academic-financial interaction
        df['academic_financial_balance'] = (
            (df['previous_grades'] / 100) * 
            (1 - df['financial_burden'] / df['financial_burden'].max())
        )
        
        # Personal-academic interaction
        df['personal_academic_fit'] = (
            df['personal_stability'] * 
            (df['academic_engagement'])
        )
        
        # Social-financial interaction
        df['social_financial_support'] = (
            df['social_score'] * 
            (1 + df['total_financial_support'] / 10000)
        )
        
        # Overall risk score (combination of all domains)
        df['overall_risk_score'] = (
            (1 - df['academic_engagement']) * 0.3 + 
            (df['financial_burden'] / df['financial_burden'].max()) * 0.3 + 
            (1 - df['personal_stability']) * 0.2 + 
            (3 - df['social_score']) / 3 * 0.2
        )
        
        return df
    
    def engineer_features(self, df):
        """
        Apply all feature engineering steps.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with all engineered features.
        """
        logger.info("Starting feature engineering process")
        
        # Apply all feature engineering methods
        df = self.create_academic_features(df)
        df = self.create_financial_features(df)
        df = self.create_personal_features(df)
        df = self.create_demographic_features(df)
        df = self.create_interaction_features(df)
        
        logger.info(f"Feature engineering completed. New shape: {df.shape}")
        
        return df
    
    def create_feature_pipeline(self):
        """
        Create a scikit-learn pipeline for feature engineering.
        
        Returns:
            Pipeline: Feature engineering pipeline.
        """
        logger.info("Creating feature engineering pipeline")
        
        self.feature_pipeline = Pipeline([
            ('feature_engineering', FunctionTransformer(self.engineer_features))
        ])
        
        return self.feature_pipeline
    
    def save_pipeline(self, filepath):
        """
        Save the feature engineering pipeline to a file.
        
        Args:
            filepath (str): Path to save the pipeline.
        """
        import joblib
        logger.info(f"Saving feature engineering pipeline to {filepath}")
        joblib.dump(self.feature_pipeline, filepath)
    
    def load_pipeline(self, filepath):
        """
        Load a feature engineering pipeline from a file.
        
        Args:
            filepath (str): Path to the saved pipeline.
            
        Returns:
            Pipeline: Loaded pipeline.
        """
        import joblib
        logger.info(f"Loading feature engineering pipeline from {filepath}")
        self.feature_pipeline = joblib.load(filepath)
        return self.feature_pipeline


def main():
    """
    Example usage of the FeatureEngineer class.
    """
    # Generate sample data for testing
    from src.data.generate_sample_data import generate_sample_data
    
    # Generate sample data
    data = generate_sample_data(num_students=100, output_dir='data')
    df = data['combined']
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Apply feature engineering
    df_engineered = feature_engineer.engineer_features(df.copy())
    
    # Create and save pipeline
    feature_engineer.create_feature_pipeline()
    os.makedirs('models', exist_ok=True)
    feature_engineer.save_pipeline('models/feature_engineering_pipeline.joblib')
    
    # Display results
    print("Original features:", len(df.columns))
    print("Engineered features:", len(df_engineered.columns))
    print("New features added:", len(df_engineered.columns) - len(df.columns))
    
    # Show some of the new features
    new_features = [col for col in df_engineered.columns if col not in df.columns]
    print("\nSample of new features:")
    for feature in new_features[:5]:
        print(f"- {feature}")


if __name__ == "__main__":
    main()