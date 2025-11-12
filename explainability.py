import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
from typing import Dict, List, Tuple, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelExplainer:
    """
    A class for explaining model predictions using SHAP and LIME.
    Provides both global (overall) and local (per-student) explanations.
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.feature_names = None
        self.shap_explainer = None
        self.lime_explainer = None
        self.X_train = None
        self.X_test = None
        
    def load_model_and_data(self, model_path, X_train, X_test, preprocessor_path=None, feature_engineer_path=None):
        """
        Load the trained model and data for explanation.
        
        Args:
            model_path (str): Path to the saved model.
            X_train (pd.DataFrame): Training data for SHAP explainer.
            X_test (pd.DataFrame): Test data for explanations.
            preprocessor_path (str): Path to the saved preprocessor.
            feature_engineer_path (str): Path to the saved feature engineer.
        """
        logger.info("Loading model and data for explanation")
        
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
        
        # Store data
        self.X_train = X_train
        self.X_test = X_test
        
        # Initialize explainers
        self._initialize_shap_explainer()
        self._initialize_lime_explainer()
    
    def _initialize_shap_explainer(self):
        """
        Initialize SHAP explainer based on model type.
        """
        logger.info("Initializing SHAP explainer")
        
        try:
            # Convert to numpy array if needed
            X_train_array = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
            
            # Choose appropriate explainer based on model type
            model_type = type(self.model).__name__
            
            if model_type in ['RandomForestClassifier', 'GradientBoostingClassifier', 'DecisionTreeClassifier', 'XGBClassifier']:
                self.shap_explainer = shap.TreeExplainer(self.model, X_train_array)
            elif model_type in ['LogisticRegression', 'LinearRegression']:
                self.shap_explainer = shap.LinearExplainer(self.model, X_train_array)
            else:
                # Use KernelExplainer as a fallback for any model type
                self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, X_train_array[:100])  # Use subset for performance
            
            logger.info(f"SHAP explainer initialized using {model_type}")
            
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {str(e)}")
            self.shap_explainer = None
    
    def _initialize_lime_explainer(self):
        """
        Initialize LIME explainer.
        """
        logger.info("Initializing LIME explainer")
        
        try:
            # Convert to numpy array if needed
            X_train_array = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
            
            # Get feature names
            feature_names = self.feature_names if self.feature_names else [f'feature_{i}' for i in range(X_train_array.shape[1])]
            
            # Get class names
            class_names = ['No Dropout', 'Dropout']
            
            # Initialize LIME explainer
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train_array,
                feature_names=feature_names,
                class_names=class_names,
                mode='classification'
            )
            
            logger.info("LIME explainer initialized")
            
        except Exception as e:
            logger.error(f"Error initializing LIME explainer: {str(e)}")
            self.lime_explainer = None
    
    def get_global_feature_importance(self, method='shap', top_n=20):
        """
        Get global feature importance using SHAP or LIME.
        
        Args:
            method (str): Method to use ('shap' or 'lime').
            top_n (int): Number of top features to return.
            
        Returns:
            pd.DataFrame: DataFrame with feature importance.
        """
        logger.info(f"Calculating global feature importance using {method}")
        
        if method == 'shap' and self.shap_explainer:
            return self._get_shap_global_importance(top_n)
        elif method == 'lime' and self.lime_explainer:
            return self._get_lime_global_importance(top_n)
        else:
            logger.error(f"Explainer not available for method: {method}")
            return pd.DataFrame()
    
    def _get_shap_global_importance(self, top_n=20):
        """
        Get global feature importance using SHAP.
        
        Args:
            top_n (int): Number of top features to return.
            
        Returns:
            pd.DataFrame: DataFrame with feature importance.
        """
        logger.info("Calculating SHAP global feature importance")
        
        try:
            # Calculate SHAP values
            X_test_array = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test
            shap_values = self.shap_explainer.shap_values(X_test_array)
            
            # For binary classification, get SHAP values for the positive class
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            
            # Calculate mean absolute SHAP values for each feature
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            
            # Create DataFrame with feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_shap_values
            }).sort_values('importance', ascending=False)
            
            return feature_importance.head(top_n)
            
        except Exception as e:
            logger.error(f"Error calculating SHAP global importance: {str(e)}")
            return pd.DataFrame()
    
    def _get_lime_global_importance(self, top_n=20):
        """
        Get global feature importance using LIME.
        
        Args:
            top_n (int): Number of top features to return.
            
        Returns:
            pd.DataFrame: DataFrame with feature importance.
        """
        logger.info("Calculating LIME global feature importance")
        
        try:
            # Sample a subset of test data for efficiency
            X_test_sample = self.X_test.sample(min(100, len(self.X_test)), random_state=42)
            X_test_array = X_test_sample.values if isinstance(X_test_sample, pd.DataFrame) else X_test_sample
            
            # Aggregate feature importance across multiple samples
            feature_importance_dict = {}
            
            for i in range(len(X_test_array)):
                # Get LIME explanation for this instance
                exp = self.lime_explainer.explain_instance(
                    X_test_array[i], 
                    self.model.predict_proba,
                    num_features=len(self.feature_names)
                )
                
                # Aggregate feature importance
                for feature, importance in exp.as_list():
                    if feature in feature_importance_dict:
                        feature_importance_dict[feature].append(abs(importance))
                    else:
                        feature_importance_dict[feature] = [abs(importance)]
            
            # Calculate mean importance for each feature
            mean_importance = {feature: np.mean(importances) for feature, importances in feature_importance_dict.items()}
            
            # Create DataFrame with feature importance
            feature_importance = pd.DataFrame({
                'feature': list(mean_importance.keys()),
                'importance': list(mean_importance.values())
            }).sort_values('importance', ascending=False)
            
            return feature_importance.head(top_n)
            
        except Exception as e:
            logger.error(f"Error calculating LIME global importance: {str(e)}")
            return pd.DataFrame()
    
    def get_local_explanation(self, instance_index, method='shap', num_features=10):
        """
        Get local explanation for a specific instance.
        
        Args:
            instance_index (int): Index of the instance to explain.
            method (str): Method to use ('shap' or 'lime').
            num_features (int): Number of features to include in explanation.
            
        Returns:
            dict: Local explanation.
        """
        logger.info(f"Getting local explanation for instance {instance_index} using {method}")
        
        if method == 'shap' and self.shap_explainer:
            return self._get_shap_local_explanation(instance_index, num_features)
        elif method == 'lime' and self.lime_explainer:
            return self._get_lime_local_explanation(instance_index, num_features)
        else:
            logger.error(f"Explainer not available for method: {method}")
            return {}
    
    def _get_shap_local_explanation(self, instance_index, num_features=10):
        """
        Get local explanation using SHAP.
        
        Args:
            instance_index (int): Index of the instance to explain.
            num_features (int): Number of features to include in explanation.
            
        Returns:
            dict: Local explanation.
        """
        logger.info(f"Calculating SHAP local explanation for instance {instance_index}")
        
        try:
            # Get instance
            X_test_array = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test
            instance = X_test_array[instance_index:instance_index+1]
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(instance)
            
            # For binary classification, get SHAP values for the positive class
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            
            # Get base value (expected value)
            expected_value = self.shap_explainer.expected_value
            if isinstance(expected_value, list) and len(expected_value) == 2:
                expected_value = expected_value[1]
            
            # Create explanation dictionary
            explanation = {
                'expected_value': expected_value,
                'shap_values': shap_values[0].tolist(),
                'feature_values': instance[0].tolist(),
                'feature_names': self.feature_names,
                'top_features': []
            }
            
            # Get top features
            feature_importance = [(self.feature_names[i], shap_values[0][i]) for i in range(len(self.feature_names))]
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            explanation['top_features'] = feature_importance[:num_features]
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error calculating SHAP local explanation: {str(e)}")
            return {}
    
    def _get_lime_local_explanation(self, instance_index, num_features=10):
        """
        Get local explanation using LIME.
        
        Args:
            instance_index (int): Index of the instance to explain.
            num_features (int): Number of features to include in explanation.
            
        Returns:
            dict: Local explanation.
        """
        logger.info(f"Calculating LIME local explanation for instance {instance_index}")
        
        try:
            # Get instance
            X_test_array = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test
            instance = X_test_array[instance_index]
            
            # Get LIME explanation
            exp = self.lime_explainer.explain_instance(
                instance, 
                self.model.predict_proba,
                num_features=num_features
            )
            
            # Create explanation dictionary
            explanation = {
                'predicted_prob': self.model.predict_proba([instance])[0][1],
                'top_features': exp.as_list(),
                'intercept': exp.intercept[1],
                'intercept_pos': exp.intercept_pos,
                'intercept_neg': exp.intercept_neg,
                'score': exp.score,
                'local_pred': exp.local_pred[0]
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error calculating LIME local explanation: {str(e)}")
            return {}
    
    def plot_global_feature_importance(self, method='shap', top_n=15, save_path=None):
        """
        Plot global feature importance.
        
        Args:
            method (str): Method to use ('shap' or 'lime').
            top_n (int): Number of top features to plot.
            save_path (str): Path to save the plot.
        """
        logger.info(f"Plotting global feature importance using {method}")
        
        # Get feature importance
        feature_importance = self.get_global_feature_importance(method, top_n)
        
        if feature_importance.empty:
            logger.error("No feature importance data available")
            return
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Horizontal bar plot
        sns.barplot(x='importance', y='feature', data=feature_importance)
        
        # Set labels and title
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title(f'Global Feature Importance ({method.upper()})')
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_shap_summary(self, save_path=None):
        """
        Plot SHAP summary plot.
        
        Args:
            save_path (str): Path to save the plot.
        """
        logger.info("Plotting SHAP summary")
        
        if not self.shap_explainer:
            logger.error("SHAP explainer not available")
            return
        
        try:
            # Get SHAP values
            X_test_array = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test
            shap_values = self.shap_explainer.shap_values(X_test_array)
            
            # For binary classification, get SHAP values for the positive class
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            
            # Create summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test_array, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting SHAP summary: {str(e)}")
    
    def plot_shap_dependence(self, feature, save_path=None):
        """
        Plot SHAP dependence plot for a specific feature.
        
        Args:
            feature (str): Feature name.
            save_path (str): Path to save the plot.
        """
        logger.info(f"Plotting SHAP dependence for feature: {feature}")
        
        if not self.shap_explainer:
            logger.error("SHAP explainer not available")
            return
        
        try:
            # Get feature index
            if feature not in self.feature_names:
                logger.error(f"Feature '{feature}' not found in feature names")
                return
            
            feature_idx = self.feature_names.index(feature)
            
            # Get SHAP values
            X_test_array = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test
            shap_values = self.shap_explainer.shap_values(X_test_array)
            
            # For binary classification, get SHAP values for the positive class
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            
            # Create dependence plot
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature_idx, shap_values, X_test_array, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting SHAP dependence: {str(e)}")
    
    def plot_local_explanation(self, instance_index, method='shap', save_path=None):
        """
        Plot local explanation for a specific instance.
        
        Args:
            instance_index (int): Index of the instance to explain.
            method (str): Method to use ('shap' or 'lime').
            save_path (str): Path to save the plot.
        """
        logger.info(f"Plotting local explanation for instance {instance_index} using {method}")
        
        if method == 'shap':
            self._plot_shap_local_explanation(instance_index, save_path)
        elif method == 'lime':
            self._plot_lime_local_explanation(instance_index, save_path)
        else:
            logger.error(f"Unknown method: {method}")
    
    def _plot_shap_local_explanation(self, instance_index, save_path=None):
        """
        Plot local explanation using SHAP.
        
        Args:
            instance_index (int): Index of the instance to explain.
            save_path (str): Path to save the plot.
        """
        logger.info(f"Plotting SHAP local explanation for instance {instance_index}")
        
        if not self.shap_explainer:
            logger.error("SHAP explainer not available")
            return
        
        try:
            # Get instance
            X_test_array = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test
            instance = X_test_array[instance_index:instance_index+1]
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(instance)
            
            # For binary classification, get SHAP values for the positive class
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            
            # Create force plot
            plt.figure(figsize=(15, 3))
            shap.force_plot(
                self.shap_explainer.expected_value[1] if isinstance(self.shap_explainer.expected_value, list) else self.shap_explainer.expected_value,
                shap_values[0],
                instance[0],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting SHAP local explanation: {str(e)}")
    
    def _plot_lime_local_explanation(self, instance_index, save_path=None):
        """
        Plot local explanation using LIME.
        
        Args:
            instance_index (int): Index of the instance to explain.
            save_path (str): Path to save the plot.
        """
        logger.info(f"Plotting LIME local explanation for instance {instance_index}")
        
        if not self.lime_explainer:
            logger.error("LIME explainer not available")
            return
        
        try:
            # Get instance
            X_test_array = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test
            instance = X_test_array[instance_index]
            
            # Get LIME explanation
            exp = self.lime_explainer.explain_instance(
                instance, 
                self.model.predict_proba,
                num_features=10
            )
            
            # Plot explanation
            plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure()
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting LIME local explanation: {str(e)}")
    
    def generate_explanation_report(self, output_dir='explanation_reports'):
        """
        Generate a comprehensive explanation report.
        
        Args:
            output_dir (str): Directory to save the report.
        """
        logger.info("Generating explanation report")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get global feature importance
        shap_importance = self.get_global_feature_importance('shap')
        lime_importance = self.get_global_feature_importance('lime')
        
        # Save feature importance
        shap_importance.to_csv(os.path.join(output_dir, 'shap_feature_importance.csv'), index=False)
        lime_importance.to_csv(os.path.join(output_dir, 'lime_feature_importance.csv'), index=False)
        
        # Plot global feature importance
        self.plot_global_feature_importance('shap', save_path=os.path.join(output_dir, 'shap_global_importance.png'))
        self.plot_global_feature_importance('lime', save_path=os.path.join(output_dir, 'lime_global_importance.png'))
        
        # Plot SHAP summary
        self.plot_shap_summary(save_path=os.path.join(output_dir, 'shap_summary.png'))
        
        # Get local explanations for a few examples
        sample_indices = [0, len(self.X_test)//4, len(self.X_test)//2, 3*len(self.X_test)//4, len(self.X_test)-1]
        
        for idx in sample_indices:
            # Plot SHAP local explanation
            self.plot_local_explanation(
                idx, 
                'shap', 
                save_path=os.path.join(output_dir, f'shap_local_explanation_{idx}.png')
            )
            
            # Plot LIME local explanation
            self.plot_local_explanation(
                idx, 
                'lime', 
                save_path=os.path.join(output_dir, f'lime_local_explanation_{idx}.png')
            )
        
        # Plot SHAP dependence for top features
        top_features = shap_importance.head(5)['feature'].tolist()
        for feature in top_features:
            self.plot_shap_dependence(
                feature, 
                save_path=os.path.join(output_dir, f'shap_dependence_{feature.replace(" ", "_")}.png')
            )
        
        logger.info(f"Explanation report generated in {output_dir}")


def main():
    """
    Example usage of the ModelExplainer class.
    """
    # Import required modules
    from src.data.generate_sample_data import generate_sample_data
    from src.data.data_preprocessing import DataPreprocessor
    from src.utils.feature_engineering import FeatureEngineer
    from src.models.model_trainer import ModelTrainer
    
    # Generate sample data
    logger.info("Generating sample data")
    data = generate_sample_data(num_students=500, output_dir='data')
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
    
    # Initialize model explainer
    explainer = ModelExplainer()
    explainer.load_model_and_data(
        model_path='models/best_model.joblib',
        X_train=X_train,
        X_test=X_test,
        preprocessor_path='models/preprocessor.joblib',
        feature_engineer_path='models/feature_engineering_pipeline.joblib'
    )
    
    # Get global feature importance
    shap_importance = explainer.get_global_feature_importance('shap')
    lime_importance = explainer.get_global_feature_importance('lime')
    
    print("\nTop 5 SHAP feature importance:")
    print(shap_importance.head())
    
    print("\nTop 5 LIME feature importance:")
    print(lime_importance.head())
    
    # Get local explanation for first instance
    local_explanation = explainer.get_local_explanation(0, 'shap')
    print("\nLocal explanation for first instance (SHAP):")
    print(f"Expected value: {local_explanation.get('expected_value', 'N/A')}")
    print("Top features:")
    for feature, value in local_explanation.get('top_features', [])[:5]:
        print(f"  {feature}: {value:.4f}")
    
    # Generate explanation report
    explainer.generate_explanation_report()
    
    logger.info("Model explanation completed successfully")


if __name__ == "__main__":
    main()