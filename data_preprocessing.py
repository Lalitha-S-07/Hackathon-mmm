import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A class for preprocessing student data from multiple verification stages.
    Handles data loading, cleaning, and preprocessing for model training.
    """
    
    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []
        self.preprocessor = None
        self.label_encoders = {}
        
    def load_data(self, file_paths):
        """
        Load data from multiple verification stages.
        
        Args:
            file_paths (dict): Dictionary with stage names as keys and file paths as values.
                              Example: {'stage1': 'data/stage1.csv', 'stage2': 'data/stage2.csv'}
        
        Returns:
            pd.DataFrame: Combined dataframe with all verification stages data.
        """
        logger.info("Loading data from multiple verification stages")
        dataframes = []
        
        for stage, path in file_paths.items():
            if os.path.exists(path):
                logger.info(f"Loading data from {stage}: {path}")
                df = pd.read_csv(path)
                df['verification_stage'] = stage
                dataframes.append(df)
            else:
                logger.warning(f"File not found: {path}")
        
        if not dataframes:
            raise ValueError("No data files found. Please check file paths.")
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined data shape: {combined_df.shape}")
        
        return combined_df
    
    def identify_feature_types(self, df):
        """
        Identify numeric and categorical features in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe.
        """
        logger.info("Identifying feature types")
        
        # Exclude target column and ID columns
        exclude_cols = ['student_id', 'dropout_status', 'verification_stage']
        
        # Identify numeric features
        self.numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.numeric_features = [col for col in self.numeric_features if col not in exclude_cols]
        
        # Identify categorical features
        self.categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.categorical_features = [col for col in self.categorical_features if col not in exclude_cols]
        
        logger.info(f"Numeric features: {len(self.numeric_features)}")
        logger.info(f"Categorical features: {len(self.categorical_features)}")
    
    def clean_data(self, df):
        """
        Clean the dataframe by handling missing values and duplicates.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            
        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        logger.info("Cleaning data")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        logger.info(f"Missing values per column:\n{missing_values[missing_values > 0]}")
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        return df
    
    def create_preprocessing_pipeline(self):
        """
        Create a preprocessing pipeline for numeric and categorical features.
        
        Returns:
            ColumnTransformer: Preprocessing pipeline.
        """
        logger.info("Creating preprocessing pipeline")
        
        # Numeric pipeline: impute missing values and scale
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline: impute missing values and one-hot encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return self.preprocessor
    
    def preprocess_data(self, df, target_column='dropout_status'):
        """
        Preprocess the data for model training.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            target_column (str): Name of the target column.
            
        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector.
        """
        logger.info("Preprocessing data")
        
        # Clean data
        df = self.clean_data(df)
        
        # Identify feature types
        self.identify_feature_types(df)
        
        # Create preprocessing pipeline
        self.create_preprocessing_pipeline()
        
        # Separate features and target
        X = df.drop(columns=[target_column, 'student_id'])
        y = df[target_column]
        
        # Apply preprocessing
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing
        feature_names = []
        feature_names.extend(self.numeric_features)
        
        # Get one-hot encoded feature names
        if hasattr(self.preprocessor.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
            cat_features = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_features)
        
        # Create a dataframe with processed features
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
        
        logger.info(f"Processed data shape: {X_processed_df.shape}")
        
        return X_processed_df, y
    
    def save_preprocessor(self, filepath):
        """
        Save the fitted preprocessor to a file.
        
        Args:
            filepath (str): Path to save the preprocessor.
        """
        import joblib
        logger.info(f"Saving preprocessor to {filepath}")
        joblib.dump(self.preprocessor, filepath)
    
    def load_preprocessor(self, filepath):
        """
        Load a preprocessor from a file.
        
        Args:
            filepath (str): Path to the saved preprocessor.
            
        Returns:
            ColumnTransformer: Loaded preprocessor.
        """
        import joblib
        logger.info(f"Loading preprocessor from {filepath}")
        self.preprocessor = joblib.load(filepath)
        return self.preprocessor


def main():
    """
    Example usage of the DataPreprocessor class.
    """
    # Example file paths (replace with actual paths)
    file_paths = {
        'stage1': 'data/stage1_data.csv',
        'stage2': 'data/stage2_data.csv',
        'stage3': 'data/stage3_data.csv',
        'stage4': 'data/stage4_data.csv'
    }
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    try:
        df = preprocessor.load_data(file_paths)
        
        # Preprocess data
        X, y = preprocessor.preprocess_data(df)
        
        # Save preprocessor
        os.makedirs('models', exist_ok=True)
        preprocessor.save_preprocessor('models/preprocessor.joblib')
        
        logger.info("Data preprocessing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")


if __name__ == "__main__":
    main()