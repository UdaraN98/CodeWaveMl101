import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)
import optuna
from optuna.trial import Trial
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import json
import subprocess
import os
import time
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
from  data_preprocessing import DataPreprocessor  


@dataclass
class GitInfo:
    """Git commit information"""
    commit_hash: str
    branch: str
    author: str
    message: str
    timestamp: str


class ModelTrainer:
    """Modular class for training and evaluating ML models with MLflow and Optuna integration"""
    
    def __init__(self, data_path, test_size=0.2, random_state=42, mlflow_tracking_uri=None, experiment_name=None):
        """Initialize ModelTrainer with dataset and MLflow configuration
        
        Args:
            data_path: Path to the dataset CSV file
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
            mlflow_tracking_uri: URI for MLflow tracking server
            experiment_name: Name of the MLflow experiment
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.model_classes = {}  # Registry for model classes
        
        # MLflow setup
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name or "default_experiment"
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        # MLflow client for model registry
        self.mlflow_client = MlflowClient()
        
        # Git info
        self.git_info = self._get_git_info()
    
    def register_model(self, model_name: str, model_class, **default_params):
        """Register a model class for easy training
        
        Args:
            model_name: Name to identify the model
            model_class: Scikit-learn model class (not instance)
            **default_params: Default parameters for the model
        """
        self.model_classes[model_name] = {
            'class': model_class,
            'default_params': default_params
        }
        print(f"Model '{model_name}' registered successfully")
    
    def _get_git_info(self) -> Optional[GitInfo]:
        """Extract git information from the repository"""
        try:
            repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                cwd=repo_path, 
                text=True
            ).strip()
            
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=repo_path,
                text=True
            ).strip()
            
            author = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=format:%an'],
                cwd=repo_path,
                text=True
            ).strip()
            
            message = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=format:%s'],
                cwd=repo_path,
                text=True
            ).strip()
            
            timestamp = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=format:%ai'],
                cwd=repo_path,
                text=True
            ).strip()
            
            return GitInfo(
                commit_hash=commit_hash,
                branch=branch,
                author=author,
                message=message,
                timestamp=timestamp
            )
        except Exception as e:
            print(f"Warning: Could not retrieve git information: {e}")
            return None
    
    def _log_git_info(self):
        """Log git information to MLflow"""
        if self.git_info:
            git_info_dict = {
                'commit_hash': self.git_info.commit_hash,
                'branch': self.git_info.branch,
                'author': self.git_info.author,
                'message': self.git_info.message,
                'timestamp': self.git_info.timestamp
            }
            mlflow.log_dict(git_info_dict, 'git_info.json')
            for key, value in git_info_dict.items():
                mlflow.log_param(f'git_{key}', value)
    
    def register_model_to_registry(self, model, model_name: str, metrics: Dict[str, float],
                                   model_description: str = None, stage: str = None) -> str:
        """Register a model to MLflow Model Registry with comprehensive tagging
        
        Args:
            model: Trained scikit-learn model
            model_name: Name for the registered model
            metrics: Dictionary of performance metrics (accuracy, precision, recall, f1, auc)
            model_description: Optional description for the model
            stage: Optional stage to transition model to ('Staging', 'Production', 'Archived')
        
        Returns:
            The model version number as string
        """
        # Log general performance metrics to MLflow run
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f'final_{metric_name}', metric_value)
        
        # Infer model signature from training data
        signature = infer_signature(self.X_train, model.predict(self.X_train))
        
        # Log model with signature
        model_info = mlflow.sklearn.log_model(
            model, 
            artifact_path=model_name,
            signature=signature,
            registered_model_name=model_name
        )
        
        # Get the model version - search for the latest version instead of relying on model_info
        time.sleep(1)  # Give MLflow a moment to register the model
        
        try:
            versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            if not versions:
                print(f"Warning: No versions found for model '{model_name}'")
                return None
            
            # Get the latest version
            model_version = max([int(v.version) for v in versions])
        except Exception as e:
            print(f"Error getting model version: {e}")
            # Fallback to registered_model_version if available
            model_version = model_info.registered_model_version
            if not model_version:
                print(f"Could not determine model version for '{model_name}'")
                return None
        
        print(f"Working with model '{model_name}' version {model_version}")
        
        # Build tags dictionary
        tags = {
            'registered_at': datetime.now().isoformat(),
            'experiment_name': self.experiment_name,
        }
        
        # Add git/commit information tags
        if self.git_info:
            tags.update({
                'git_commit_hash': self.git_info.commit_hash,
                'git_branch': self.git_info.branch,
                'git_author': self.git_info.author,
                'git_commit_message': self.git_info.message[:250] if self.git_info.message else '',
                'git_commit_timestamp': self.git_info.timestamp
            })
        
        # Add performance metric tags
        for metric_name, metric_value in metrics.items():
            tags[f'metric_{metric_name}'] = str(round(metric_value, 4))
        
        # Set tags on the model version with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                for tag_key, tag_value in tags.items():
                    self.mlflow_client.set_model_version_tag(
                        name=model_name,
                        version=str(model_version),
                        key=tag_key,
                        value=str(tag_value)
                    )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Retry {attempt + 1}/{max_retries}: Waiting for model version to be available...")
                    time.sleep(2)
                else:
                    print(f"Warning: Could not set all tags after {max_retries} attempts: {e}")
        
        # Set model version description if provided
        if model_description:
            try:
                self.mlflow_client.update_model_version(
                    name=model_name,
                    version=str(model_version),
                    description=model_description
                )
            except Exception as e:
                print(f"Warning: Could not set model description: {e}")
        
        # Transition to specified stage if provided
        if stage:
            try:
                self.mlflow_client.transition_model_version_stage(
                    name=model_name,
                    version=str(model_version),
                    stage=stage
                )
                print(f"Model '{model_name}' v{model_version} transitioned to '{stage}'")
            except Exception as e:
                print(f"Warning: Could not transition model stage: {e}")
        
        print(f"Model '{model_name}' registered successfully as version {model_version}")
        print(f"  - Author: {self.git_info.author if self.git_info else 'Unknown'}")
        print(f"  - Commit: {self.git_info.commit_hash[:8] if self.git_info else 'N/A'}")
        print(f"  - Metrics: Accuracy={metrics.get('accuracy', 0):.4f}, F1={metrics.get('f1', 0):.4f}, AUC={metrics.get('auc', 0):.4f}")
        
        return str(model_version)
    
    def get_registered_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a registered model and its versions
        
        Args:
            model_name: Name of the registered model
        
        Returns:
            Dictionary with model and version information
        """
        try:
            model = self.mlflow_client.get_registered_model(model_name)
            versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            
            return {
                'name': model.name,
                'description': model.description,
                'creation_timestamp': model.creation_timestamp,
                'last_updated_timestamp': model.last_updated_timestamp,
                'versions': [
                    {
                        'version': v.version,
                        'stage': v.current_stage,
                        'status': v.status,
                        'tags': v.tags,
                        'description': v.description
                    }
                    for v in versions
                ]
            }
        except Exception as e:
            print(f"Error getting model info: {e}")
            return None
    
    def compare_model_versions(self, model_name: str) -> pd.DataFrame:
        """Compare all versions of a registered model by their metrics
        
        Args:
            model_name: Name of the registered model
        
        Returns:
            DataFrame comparing model versions
        """
        versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
        
        comparison_data = []
        for v in versions:
            row = {
                'version': v.version,
                'stage': v.current_stage,
                'status': v.status,
            }
            # Extract metric tags
            for tag_key, tag_value in v.tags.items():
                if tag_key.startswith('metric_'):
                    metric_name = tag_key.replace('metric_', '')
                    row[metric_name] = float(tag_value)
                elif tag_key in ['git_author', 'git_commit_hash', 'git_branch']:
                    row[tag_key] = tag_value
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def promote_best_model(self, model_name: str, metric: str = 'f1', stage: str = 'Production') -> str:
        """Promote the best performing model version to a specified stage
        
        Args:
            model_name: Name of the registered model
            metric: Metric to use for comparison
            stage: Stage to promote to ('Staging', 'Production')
        
        Returns:
            The version number that was promoted
        """
        comparison_df = self.compare_model_versions(model_name)
        
        if comparison_df.empty:
            print(f"No versions found for model '{model_name}'")
            return None
        
        if metric not in comparison_df.columns:
            print(f"Metric '{metric}' not found in model versions")
            return None
        
        # Find the best version
        best_idx = comparison_df[metric].idxmax()
        best_version = comparison_df.loc[best_idx, 'version']
        
        # Archive current production model if exists
        current_prod = [v for v in self.mlflow_client.search_model_versions(f"name='{model_name}'")
                        if v.current_stage == stage]
        for v in current_prod:
            self.mlflow_client.transition_model_version_stage(
                name=model_name,
                version=str(v.version),
                stage='Archived'
            )
            print(f"Archived previous {stage} model: v{v.version}")
        
        # Promote best model
        self.mlflow_client.transition_model_version_stage(
            name=model_name,
            version=str(best_version),
            stage=stage
        )
        
        print(f"Promoted model '{model_name}' v{best_version} to '{stage}' (best {metric}: {comparison_df.loc[best_idx, metric]:.4f})")
        return str(best_version)
    
    def train_model(self, model_name: str, **params):
        """Train a registered model with custom parameters
        
        Args:
            model_name: Name of the registered model
            **params: Parameters to override defaults
        """
        if model_name not in self.model_classes:
            raise ValueError(f"Model '{model_name}' not registered. Available: {list(self.model_classes.keys())}")
        
        model_config = self.model_classes[model_name]
        model_class = model_config['class']
        
        # Merge default params with provided params
        final_params = {**model_config['default_params'], **params}
        
        model = model_class(**final_params)
        model.fit(self.X_train, self.y_train)
        self.models[model_name] = model
        print(f"Model '{model_name}' trained successfully with params: {final_params}")
        return model
    
    def load_dataset(self):
        """Load and prepare dataset"""
        self.dataset = pd.read_csv(self.data_path)
        print(f"Dataset loaded with shape: {self.dataset.shape}")
        return self.dataset
    
    def split_data(self, target_column='Churn'):
        """Split dataset into train and test sets"""
        X = self.dataset.drop(columns=[target_column])
        y = self.dataset[target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        print(f"Train set size: {self.X_train.shape}, Test set size: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self, model_name='lr_default', **kwargs):
        """Train default Logistic Regression model"""
        model = LogisticRegression(**kwargs)
        model.fit(self.X_train, self.y_train)
        self.models[model_name] = model
        print(f"Model '{model_name}' trained successfully")
        return model
    
    def grid_search_logistic_regression(self, param_grid, cv=5, scoring='recall_macro'):
        """Perform GridSearchCV for Logistic Regression"""
        grid_search = GridSearchCV(
            estimator=LogisticRegression(),
            param_grid=param_grid,
            cv=cv,
            scoring=scoring
        )
        grid_search.fit(self.X_train, self.y_train)
        self.models['lr_grid_search'] = grid_search.best_estimator_
        print(f"Best parameters (GridSearchCV): {grid_search.best_params_}")
        return grid_search
    
    def random_search_logistic_regression(self, param_dist, n_iter=10, cv=5, scoring='recall_macro'):
        """Perform RandomizedSearchCV for Logistic Regression"""
        random_search = RandomizedSearchCV(
            estimator=LogisticRegression(),
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=self.random_state
        )
        random_search.fit(self.X_train, self.y_train)
        self.models['lr_random_search'] = random_search.best_estimator_
        print(f"Best parameters (RandomizedSearchCV): {random_search.best_params_}")
        return random_search
    
    def optuna_optimization(self, objective_func: Callable, n_trials: int = 10, 
                           model_name: str = 'optuna_model', log_to_mlflow: bool = True) -> tuple:
        """Optimize hyperparameters using Optuna with comprehensive metrics logging
        
        Args:
            objective_func: Callable that takes a Trial and returns a score
            n_trials: Number of trials to run
            model_name: Name to store the best model
            log_to_mlflow: Whether to log results to MLflow
        
        Returns:
            Tuple of (study, best_model)
        """
        study = optuna.create_study(direction='maximize')
        
        # Wrapper to log metrics
        def objective_with_logging(trial: Trial):
            score = objective_func(trial)
            
            # Log metrics to MLflow
            if log_to_mlflow:
                mlflow.log_metric('optuna_score', score, step=trial.number)
                
                # Log trial parameters
                for param_name, param_value in trial.params.items():
                    mlflow.log_param(f'trial_{trial.number}_{param_name}', param_value)
            
            return score
        
        study.optimize(objective_with_logging, n_trials=n_trials)
        
        # Get best model and log comprehensive metrics
        best_model = None
        if hasattr(study, 'best_trial') and study.best_trial is not None:
            print(f"Best parameters (Optuna): {study.best_params}")
            print(f"Best score: {study.best_value}")
            
            if log_to_mlflow:
                mlflow.log_param('optuna_best_params', str(study.best_params))
                mlflow.log_metric('optuna_best_score', study.best_value)
        
        return study, best_model
    
    def optuna_optimization_with_model(self, model_class, param_distributions: Dict[str, Any],
                                       n_trials: int = 10, model_name: str = 'optuna_model',
                                       log_to_mlflow: bool = True) -> tuple:
        """Optimize and train a model using Optuna with full metrics logging
        
        Args:
            model_class: Scikit-learn model class
            param_distributions: Dictionary mapping parameter names to optuna suggest calls
            n_trials: Number of trials
            model_name: Name to store the best model
            log_to_mlflow: Whether to log to MLflow
        
        Returns:
            Tuple of (study, best_model)
        """
        def objective(trial: Trial):
            # Suggest parameters using the provided distributions
            params = {}
            for param_name, suggest_func in param_distributions.items():
                params[param_name] = suggest_func(trial)
            
            # Create and train model
            model = model_class(**params)
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Calculate all metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            
            try:
                auc = roc_auc_score(self.y_test, y_pred_proba)
            except:
                auc = 0.0
            
            # Log all metrics to Optuna (MLflow logging happens in callback)
            trial.set_user_attr('accuracy', accuracy)
            trial.set_user_attr('precision', precision)
            trial.set_user_attr('recall', recall)
            trial.set_user_attr('f1', f1)
            trial.set_user_attr('auc', auc)
            
            if log_to_mlflow:
                step = trial.number
                mlflow.log_metric('trial_accuracy', accuracy, step=step)
                mlflow.log_metric('trial_precision', precision, step=step)
                mlflow.log_metric('trial_recall', recall, step=step)
                mlflow.log_metric('trial_f1', f1, step=step)
                mlflow.log_metric('trial_auc', auc, step=step)
                
                # Log parameters
                for param_name, param_value in params.items():
                    mlflow.log_param(f'trial_{step}_{param_name}', param_value)
            
            # Return F1 as the optimization metric
            return f1
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Train final model with best parameters
        best_params = study.best_params
        best_model = model_class(**best_params)
        best_model.fit(self.X_train, self.y_train)
        self.models[model_name] = best_model
        
        print(f"Best parameters (Optuna): {best_params}")
        print(f"Best trial number: {study.best_trial.number}")
        
        # Print trial statistics
        best_trial = study.best_trial
        print(f"\nBest Trial Metrics:")
        print(f"  Accuracy: {best_trial.user_attrs.get('accuracy', 'N/A'):.4f}")
        print(f"  Precision: {best_trial.user_attrs.get('precision', 'N/A'):.4f}")
        print(f"  Recall: {best_trial.user_attrs.get('recall', 'N/A'):.4f}")
        print(f"  F1 Score: {best_trial.user_attrs.get('f1', 'N/A'):.4f}")
        print(f"  AUC: {best_trial.user_attrs.get('auc', 'N/A'):.4f}")
        
        return study, best_model
    
    def evaluate_model(self, model_name, y_pred=None, model=None, log_to_mlflow=True):
        """Evaluate model performance with comprehensive metrics
        
        Args:
            model_name: Name of the model
            y_pred: Pre-computed predictions (optional)
            model: Model instance (optional)
            log_to_mlflow: Whether to log metrics to MLflow
        """
        if y_pred is None and model is not None:
            y_pred = model.predict(self.X_test)
        elif y_pred is None:
            y_pred = self.models[model_name].predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        try:
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            auc = roc_auc_score(self.y_test, y_pred_proba)
        except:
            auc = 0.0
        
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        # Log to MLflow
        if log_to_mlflow:
            # Log with model name prefix
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f'{model_name}_{metric_name}', metric_value)
            
            # Also log general metrics for easy comparison
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
        
        return metrics
    
    def plot_confusion_matrix(self, model_name, y_pred=None, model=None, figsize=(6, 4)):
        """Visualize confusion matrix"""
        if y_pred is None and model is not None:
            y_pred = model.predict(self.X_test)
        elif y_pred is None:
            y_pred = self.models[model_name].predict(self.X_test)
        
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'], 
                    yticklabels=['No Churn', 'Churn'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.show()


def main():
    """Main execution function demonstrating enhanced ModelTrainer"""
    # Get the project root directory dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'Data')
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # run the data preprocessing script
    preprocessor = DataPreprocessor(
        input_dir=data_dir,
        output_dir=data_dir
    )
    preprocessor.load_data()
    preprocessor.apply_preprocessing()
    preprocessor.save_data()    

    # Initialize trainer with MLflow
    data_path = os.path.join(data_dir, 'customer_churn_dataset_prepared.csv')
    
    # Get MLflow tracking URI from environment or use local
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', None)
    if mlflow_uri:
        print(f"Using remote MLflow tracking server: {mlflow_uri}")
    else:
        print("Using local MLflow tracking")
    
    trainer = ModelTrainer(
        data_path,
        mlflow_tracking_uri=mlflow_uri,
        experiment_name='customer_churn_optimization'
    )
    
    # Load and prepare data
    trainer.load_dataset()
    trainer.split_data()
    
    # Register models for easy training
    trainer.register_model(
        'logistic_regression',
        LogisticRegression,
        random_state=42,
        max_iter=1000
    )
    
    trainer.register_model(
        'decision_tree',
        DecisionTreeClassifier,
        random_state=42
    )
    
    # ==================== Train Individual Models ====================
    print("\n" + "="*60)
    print("Training Individual Models")
    print("="*60)
    
    # Train Logistic Regression
    with mlflow.start_run(run_name='logistic_regression_baseline'):
        trainer._log_git_info()
        lr_model = trainer.train_model('logistic_regression')
        lr_metrics = trainer.evaluate_model('logistic_regression', model=lr_model)
        
        # Register model to registry with tags
        trainer.register_model_to_registry(
            model=lr_model,
            model_name='churn_logistic_regression',
            metrics=lr_metrics,
            model_description='Baseline Logistic Regression model for customer churn prediction',
            stage='Staging'
        )
    
    # Train Decision Tree
    with mlflow.start_run(run_name='decision_tree_baseline'):
        trainer._log_git_info()
        dt_model = trainer.train_model('decision_tree')
        dt_metrics = trainer.evaluate_model('decision_tree', model=dt_model)
        
        # Register model to registry with tags
        trainer.register_model_to_registry(
            model=dt_model,
            model_name='churn_decision_tree',
            metrics=dt_metrics,
            model_description='Baseline Decision Tree model for customer churn prediction',
            stage='Staging'
        )
    
    # ==================== Optuna Hyperparameter Optimization ====================
    print("\n" + "="*60)
    print("Optuna Hyperparameter Optimization for Logistic Regression")
    print("="*60)
    
    param_distributions_lr = {
        'C': lambda trial: trial.suggest_float('C', 0.001, 100.0, log=True),
        'max_iter': lambda trial: trial.suggest_int('max_iter', 100, 500),
        'solver': lambda trial: trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg'])
    }
    
    with mlflow.start_run(run_name='logistic_regression_optuna'):
        trainer._log_git_info()
        mlflow.log_param('optuna_n_trials', 20)
        
        study_lr, best_lr_model = trainer.optuna_optimization_with_model(
            LogisticRegression,
            param_distributions_lr,
            n_trials=20,
            model_name='lr_optuna',
            log_to_mlflow=True
        )
        
        # Evaluate the best model
        best_lr_metrics = trainer.evaluate_model('lr_optuna', model=best_lr_model, log_to_mlflow=True)
        
        # Log study details
        mlflow.log_param('optuna_best_params', str(study_lr.best_params))
        mlflow.log_metric('optuna_best_score', study_lr.best_value)
        
        # Register optimized model to registry
        trainer.register_model_to_registry(
            model=best_lr_model,
            model_name='churn_logistic_regression',
            metrics=best_lr_metrics,
            model_description=f'Optuna-optimized Logistic Regression. Best params: {study_lr.best_params}',
            stage='Staging'
        )
    
    # ==================== Optuna Hyperparameter Optimization for Decision Tree ====================
    print("\n" + "="*60)
    print("Optuna Hyperparameter Optimization for Decision Tree")
    print("="*60)
    
    param_distributions_dt = {
        'max_depth': lambda trial: trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': lambda trial: trial.suggest_int('min_samples_leaf', 1, 10),
        'criterion': lambda trial: trial.suggest_categorical('criterion', ['gini', 'entropy'])
    }
    
    with mlflow.start_run(run_name='decision_tree_optuna'):
        trainer._log_git_info()
        mlflow.log_param('optuna_n_trials', 20)
        
        study_dt, best_dt_model = trainer.optuna_optimization_with_model(
            DecisionTreeClassifier,
            param_distributions_dt,
            n_trials=20,
            model_name='dt_optuna',
            log_to_mlflow=True
        )
        
        # Evaluate the best model
        best_dt_metrics = trainer.evaluate_model('dt_optuna', model=best_dt_model, log_to_mlflow=True)
        
        # Log study details
        mlflow.log_param('optuna_best_params', str(study_dt.best_params))
        mlflow.log_metric('optuna_best_score', study_dt.best_value)
        
        # Register optimized model to registry
        trainer.register_model_to_registry(
            model=best_dt_model,
            model_name='churn_decision_tree',
            metrics=best_dt_metrics,
            model_description=f'Optuna-optimized Decision Tree. Best params: {study_dt.best_params}',
            stage='Staging'
        )
    
    # ==================== Model Comparison and Promotion ====================
    print("\n" + "="*60)
    print("Model Version Comparison")
    print("="*60)
    
    # Compare Logistic Regression versions
    print("\nLogistic Regression Model Versions:")
    lr_comparison = trainer.compare_model_versions('churn_logistic_regression')
    print(lr_comparison.to_string())
    
    # Compare Decision Tree versions
    print("\nDecision Tree Model Versions:")
    dt_comparison = trainer.compare_model_versions('churn_decision_tree')
    print(dt_comparison.to_string())
    
    # Promote best models to Production
    print("\n" + "="*60)
    print("Promoting Best Models to Production")
    print("="*60)
    
    trainer.promote_best_model('churn_logistic_regression', metric='f1', stage='Production')
    trainer.promote_best_model('churn_decision_tree', metric='f1', stage='Production')
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("Check MLflow UI for detailed experiment tracking and model registry")
    print("="*60)


if __name__ == '__main__':
    main()