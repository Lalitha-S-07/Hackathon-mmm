import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import xgboost as xgb
import joblib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    A class for training and evaluating multiple machine learning models
    for student dropout prediction.
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.model_scores = {}
        self.feature_names = None
        
    def initialize_models(self):
        """
        Initialize a dictionary of machine learning models to train.
        """
        logger.info("Initializing models")
        
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """
        Split data into training, validation, and test sets.
        
        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
            test_size (float): Proportion of data for test set.
            val_size (float): Proportion of training data for validation set.
            random_state (int): Random seed for reproducibility.
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Splitting data into train, validation, and test sets")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation sets
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def evaluate_model(self, model, X_train, X_val, y_train, y_val, model_name):
        """
        Evaluate a single model on validation data.
        
        Args:
            model: Trained model object.
            X_train (pd.DataFrame): Training features.
            X_val (pd.DataFrame): Validation features.
            y_train (pd.Series): Training target.
            y_val (pd.Series): Validation target.
            model_name (str): Name of the model.
            
        Returns:
            dict: Dictionary of evaluation metrics.
        """
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        logger.info(f"{model_name} - ROC AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def train_models(self, X_train, X_val, y_train, y_val):
        """
        Train all initialized models and evaluate their performance.
        
        Args:
            X_train (pd.DataFrame): Training features.
            X_val (pd.DataFrame): Validation features.
            y_train (pd.Series): Training target.
            y_val (pd.Series): Validation target.
        """
        logger.info("Training and evaluating models")
        
        self.feature_names = X_train.columns.tolist()
        
        for name, model in self.models.items():
            logger.info(f"Training {name}")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate the model
            metrics = self.evaluate_model(model, X_train, X_val, y_train, y_val, name)
            self.model_scores[name] = metrics
        
        # Select best model based on ROC AUC
        self.best_model_name = max(self.model_scores, key=lambda x: self.model_scores[x]['roc_auc'])
        self.best_model = self.models[self.best_model_name]
        
        logger.info(f"Best model: {self.best_model_name} with ROC AUC: {self.model_scores[self.best_model_name]['roc_auc']:.4f}")
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='Random Forest'):
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            model_name (str): Name of the model to tune.
            
        Returns:
            dict: Best parameters and best score.
        """
        logger.info(f"Performing hyperparameter tuning for {model_name}")
        
        # Define parameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return None
        
        # Get the model
        model = self.models.get(model_name)
        if model is None:
            logger.error(f"Model {model_name} not found")
            return None
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, 
            param_grids[model_name], 
            cv=5, 
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    
    def evaluate_on_test_set(self, X_test, y_test):
        """
        Evaluate the best model on the test set.
        
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target.
            
        Returns:
            dict: Test set evaluation metrics.
        """
        logger.info("Evaluating best model on test set")
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Test set performance - ROC AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def save_model(self, filepath):
        """
        Save the best trained model to a file.
        
        Args:
            filepath (str): Path to save the model.
        """
        logger.info(f"Saving best model to {filepath}")
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'model_scores': self.model_scores
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath (str): Path to the saved model.
            
        Returns:
            dict: Loaded model data.
        """
        logger.info(f"Loading model from {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        self.model_scores = model_data['model_scores']
        
        return model_data
    
    def generate_model_comparison_report(self, output_dir='models'):
        """
        Generate a comparison report of all trained models.
        
        Args:
            output_dir (str): Directory to save the report.
        """
        logger.info("Generating model comparison report")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(self.model_scores).T
        comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
        
        # Save comparison table
        os.makedirs(output_dir, exist_ok=True)
        comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot ROC AUC scores
        plt.subplot(2, 2, 1)
        comparison_df['roc_auc'].plot(kind='bar')
        plt.title('Model Comparison - ROC AUC')
        plt.ylabel('ROC AUC Score')
        plt.xticks(rotation=45)
        
        # Plot F1 scores
        plt.subplot(2, 2, 2)
        comparison_df['f1_score'].plot(kind='bar')
        plt.title('Model Comparison - F1 Score')
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45)
        
        # Plot accuracy scores
        plt.subplot(2, 2, 3)
        comparison_df['accuracy'].plot(kind='bar')
        plt.title('Model Comparison - Accuracy')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # Plot precision and recall
        plt.subplot(2, 2, 4)
        comparison_df[['precision', 'recall']].plot(kind='bar')
        plt.title('Model Comparison - Precision & Recall')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison report saved to {output_dir}")


def main():
    """
    Example usage of the ModelTrainer class.
    """
    # Import required modules
    from src.data.generate_sample_data import generate_sample_data
    from src.data.data_preprocessing import DataPreprocessor
    from src.utils.feature_engineering import FeatureEngineer
    
    # Generate sample data
    logger.info("Generating sample data")
    data = generate_sample_data(num_students=1000, output_dir='data')
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
    
    # Initialize model trainer
    trainer = ModelTrainer()
    trainer.initialize_models()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X_engineered, y)
    
    # Train models
    trainer.train_models(X_train, X_val, y_train, y_val)
    
    # Hyperparameter tuning for best model
    trainer.hyperparameter_tuning(X_train, y_train, trainer.best_model_name)
    
    # Re-train with best parameters
    trainer.train_models(X_train, X_val, y_train, y_val)
    
    # Evaluate on test set
    test_results = trainer.evaluate_on_test_set(X_test, y_test)
    
    # Generate comparison report
    trainer.generate_model_comparison_report()
    
    # Save best model
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/best_model.joblib')
    
    logger.info("Model training completed successfully")


if __name__ == "__main__":
    main()