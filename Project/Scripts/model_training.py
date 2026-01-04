import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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
import subprocess
import os
import time
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
from data_preprocessing import DataPreprocessor


@dataclass
class GitInfo:
    """Git commit information"""
    commit_hash: str
    branch: str
    author: str
    message: str
    timestamp: str


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'auc': self.auc
        }


class ModelTrainer:
    """Optimized model training and evaluation with MLflow integrations"""
    
    def __init__(self, data_path, test_size=0.2, random_state=42, 
                 mlflow_tracking_uri=None, experiment_name=None):
        """Initialize ModelTrainer with dataset and MLflow configuration"""
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.model_classes = {}
        self.trained_models_metrics = {}  # Track all trained models for comparison
        
        # MLflow setup
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name or "default_experiment"
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.mlflow_client = MlflowClient()
        
        # Git info
        self.git_info = self._get_git_info()
    
    def register_model(self, model_name: str, model_class, **default_params):
        """Register a model class for training"""
        self.model_classes[model_name] = {
            'class': model_class,
            'default_params': default_params
        }
        print(f"Model '{model_name}' registered")
    
    def _get_git_info(self) -> Optional[GitInfo]:
        """Extract git information from the repository"""
        try:
            repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                cwd=repo_path, text=True
            ).strip()
            
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=repo_path, text=True
            ).strip()
            
            author = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=format:%an'],
                cwd=repo_path, text=True
            ).strip()
            
            message = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=format:%s'],
                cwd=repo_path, text=True
            ).strip()
            
            timestamp = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=format:%ai'],
                cwd=repo_path, text=True
            ).strip()
            
            return GitInfo(commit_hash, branch, author, message, timestamp)
        except Exception as e:
            print(f"Warning: Could not retrieve git information: {e}")
            return None
    
    def _log_git_info(self):
        """Log git information to MLflow"""
        if self.git_info:
            git_params = {
                'git_commit_hash': self.git_info.commit_hash,
                'git_branch': self.git_info.branch,
                'git_author': self.git_info.author
            }
            for key, value in git_params.items():
                mlflow.log_param(key, value)
    
    def _calculate_metrics(self, model, model_name: str) -> ModelMetrics:
        """Calculate all evaluation metrics for a model"""
        y_pred = model.predict(self.X_test)
        
        # Calculate core metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        # Calculate AUC if probability predictions available
        try:
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(self.y_test, y_pred_proba)
        except:
            auc = 0.0
        
        return ModelMetrics(accuracy, precision, recall, f1, auc)
    
    def _log_metrics_to_mlflow(self, metrics: ModelMetrics):
        """Log metrics to MLflow (called once per run)"""
        for metric_name, metric_value in metrics.to_dict().items():
            mlflow.log_metric(metric_name, metric_value)
    
    def register_model_to_registry(self, model, model_name: str, registry_name: str,
                                   metrics: ModelMetrics, model_description: str = None,
                                   stage: str = 'Staging') -> str:
        """Register a model to MLflow Model Registry with minimal tags"""
        # Infer signature
        signature = infer_signature(self.X_train, model.predict(self.X_train))
        
        # Log model
        model_info = mlflow.sklearn.log_model(
            model, 
            artifact_path=model_name,
            signature=signature,
            registered_model_name=registry_name
        )
        
        # Wait for registration
        time.sleep(1)
        
        try:
            versions = self.mlflow_client.search_model_versions(f"name='{registry_name}'")
            if not versions:
                print(f"Warning: No versions found for model '{registry_name}'")
                return None
            model_version = max([int(v.version) for v in versions])
        except Exception as e:
            print(f"Error getting model version: {e}")
            return None
        
        # Set essential tags only
        tags = {
            'registered_at': datetime.now().isoformat(),
            'f1_score': str(round(metrics.f1, 4)),
            'accuracy': str(round(metrics.accuracy, 4)),
            'auc': str(round(metrics.auc, 4))
        }
        
        if self.git_info:
            tags['git_commit'] = self.git_info.commit_hash[:8]
            tags['git_author'] = self.git_info.author
        
        # Set tags with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                for tag_key, tag_value in tags.items():
                    self.mlflow_client.set_model_version_tag(
                        name=registry_name,
                        version=str(model_version),
                        key=tag_key,
                        value=str(tag_value)
                    )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"Warning: Could not set tags: {e}")
        
        # Set description
        if model_description:
            try:
                self.mlflow_client.update_model_version(
                    name=registry_name,
                    version=str(model_version),
                    description=model_description
                )
            except Exception as e:
                print(f"Warning: Could not set description: {e}")
        
        # Set stage
        try:
            self.mlflow_client.transition_model_version_stage(
                name=registry_name,
                version=str(model_version),
                stage=stage
            )
        except Exception as e:
            print(f"Warning: Could not set stage: {e}")
        
        print(f"✓ Model '{registry_name}' v{model_version} registered ({stage})")
        print(f"  F1: {metrics.f1:.4f} | Accuracy: {metrics.accuracy:.4f} | AUC: {metrics.auc:.4f}")
        
        return str(model_version)
    
    def train_model(self, model_name: str, **params):
        """Train a registered model with custom parameters"""
        if model_name not in self.model_classes:
            raise ValueError(f"Model '{model_name}' not registered")
        
        model_config = self.model_classes[model_name]
        final_params = {**model_config['default_params'], **params}
        
        model = model_config['class'](**final_params)
        model.fit(self.X_train, self.y_train)
        self.models[model_name] = model
        return model
    
    def load_dataset(self):
        """Load dataset"""
        self.dataset = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.dataset.shape}")
        return self.dataset
    
    def split_data(self, target_column='Churn'):
        """Split dataset into train and test sets"""
        X = self.dataset.drop(columns=[target_column])
        y = self.dataset[target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        print(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def optuna_optimization_with_model(self, model_class, param_distributions: Dict[str, Any],
                                       n_trials: int = 10, model_name: str = 'optuna_model') -> tuple:
        """Optimize hyperparameters using Optuna"""
        def objective(trial: Trial):
            # Suggest parameters
            params = {name: suggest_func(trial) for name, suggest_func in param_distributions.items()}
            
            # Train model
            model = model_class(**params)
            model.fit(self.X_train, self.y_train)
            
            # Calculate F1 score for optimization
            y_pred = model.predict(self.X_test)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            
            return f1
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Train final model with best parameters
        best_model = model_class(**study.best_params)
        best_model.fit(self.X_train, self.y_train)
        self.models[model_name] = best_model
        
        print(f"\nBest params: {study.best_params}")
        print(f"Best F1 score: {study.best_value:.4f}")
        
        return study, best_model
    
    def evaluate_model(self, model_name: str, model=None, print_results: bool = True) -> ModelMetrics:
        """Evaluate model and return metrics"""
        if model is None:
            model = self.models[model_name]
        
        metrics = self._calculate_metrics(model, model_name)
        
        if print_results:
            y_pred = model.predict(self.X_test)
            print(f"\n{'='*50}")
            print(f"Model: {model_name}")
            print(f"{'='*50}")
            print(f"Accuracy:  {metrics.accuracy:.4f}")
            print(f"Precision: {metrics.precision:.4f}")
            print(f"Recall:    {metrics.recall:.4f}")
            print(f"F1 Score:  {metrics.f1:.4f}")
            print(f"AUC:       {metrics.auc:.4f}")
            print("\nConfusion Matrix:")
            print(confusion_matrix(self.y_test, y_pred))
        
        return metrics
    
    def compare_all_models(self) -> pd.DataFrame:
        """Compare all trained models and return sorted by F1 score"""
        comparison_data = []
        
        for model_name, metrics in self.trained_models_metrics.items():
            comparison_data.append({
                'model': model_name,
                'f1': metrics.f1,
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'auc': metrics.auc
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('f1', ascending=False).reset_index(drop=True)
        return df
    
    def promote_best_model_across_all(self, registry_names: list, metric: str = 'f1') -> Dict[str, str]:
        """Promote the single best model to Production, stage all others"""
        all_models = []
        
        # Collect all model versions
        for registry_name in registry_names:
            try:
                versions = self.mlflow_client.search_model_versions(f"name='{registry_name}'")
                for v in versions:
                    metric_tag = f'metric_{metric}' if f'metric_{metric}' in v.tags else metric
                    if metric_tag in v.tags:
                        all_models.append({
                            'registry_name': registry_name,
                            'version': v.version,
                            'metric_value': float(v.tags[metric_tag]),
                            'stage': v.current_stage
                        })
            except Exception as e:
                print(f"Warning: Could not retrieve versions for {registry_name}: {e}")
        
        if not all_models:
            print("No models found to promote")
            return {}
        
        # Find best model
        best_model = max(all_models, key=lambda x: x['metric_value'])
        
        print(f"\n{'='*60}")
        print("Model Promotion Summary")
        print(f"{'='*60}")
        
        results = {}
        
        # Process each model
        for model in all_models:
            registry_name = model['registry_name']
            version = model['version']
            
            if (model['registry_name'] == best_model['registry_name'] and 
                model['version'] == best_model['version']):
                # Promote best to Production
                try:
                    # Archive current production models
                    current_prod = [v for v in self.mlflow_client.search_model_versions(f"name='{registry_name}'")
                                  if v.current_stage == 'Production']
                    for v in current_prod:
                        if str(v.version) != str(version):
                            self.mlflow_client.transition_model_version_stage(
                                name=registry_name,
                                version=str(v.version),
                                stage='Archived'
                            )
                    
                    self.mlflow_client.transition_model_version_stage(
                        name=registry_name,
                        version=str(version),
                        stage='Production'
                    )
                    results[f"{registry_name}_v{version}"] = 'Production'
                    print(f"✓ PRODUCTION: {registry_name} v{version} ({metric}={model['metric_value']:.4f}) ⭐")
                except Exception as e:
                    print(f"✗ Error promoting {registry_name} v{version}: {e}")
            else:
                # Stage all others
                if model['stage'] == 'Production':
                    try:
                        self.mlflow_client.transition_model_version_stage(
                            name=registry_name,
                            version=str(version),
                            stage='Staging'
                        )
                        results[f"{registry_name}_v{version}"] = 'Staging'
                        print(f"  Staging:    {registry_name} v{version} ({metric}={model['metric_value']:.4f})")
                    except Exception as e:
                        print(f"✗ Error staging {registry_name} v{version}: {e}")
        
        print(f"{'='*60}\n")
        return results


def main():
    """Main execution function with optimized workflow"""
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Preprocessing
    preprocessor = DataPreprocessor(input_dir=data_dir, output_dir=data_dir)
    preprocessor.load_data()
    preprocessor.apply_preprocessing()
    preprocessor.save_data()
    
    # Initialize trainer
    data_path = os.path.join(data_dir, 'customer_churn_dataset_prepared.csv')
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', None)
    
    trainer = ModelTrainer(
        data_path,
        mlflow_tracking_uri=mlflow_uri,
        experiment_name='customer_churn_optimization'
    )
    
    # Load data
    trainer.load_dataset()
    trainer.split_data()
    
    # Register models
    trainer.register_model('logistic_regression', LogisticRegression, random_state=42, max_iter=1000)
    trainer.register_model('decision_tree', DecisionTreeClassifier, random_state=42)
    
    registry_names = []
    
    # ==================== Train and Optimize Models ====================
    print("\n" + "="*60)
    print("Training Models with Optuna Optimization")
    print("="*60)
    
    # Logistic Regression
    param_dist_lr = {
        'C': lambda trial: trial.suggest_float('C', 0.001, 100.0, log=True),
        'max_iter': lambda trial: trial.suggest_int('max_iter', 100, 500),
        'solver': lambda trial: trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg'])
    }
    
    with mlflow.start_run(run_name='logistic_regression_optuna'):
        trainer._log_git_info()
        mlflow.log_param('n_trials', 20)
        
        study_lr, best_lr = trainer.optuna_optimization_with_model(
            LogisticRegression, param_dist_lr, n_trials=20, model_name='lr_optuna'
        )
        
        lr_metrics = trainer.evaluate_model('lr_optuna', model=best_lr)
        trainer._log_metrics_to_mlflow(lr_metrics)
        trainer.trained_models_metrics['logistic_regression'] = lr_metrics
        
        trainer.register_model_to_registry(
            model=best_lr,
            model_name='lr_optuna',
            registry_name='churn_predictor_lr',
            metrics=lr_metrics,
            model_description=f'Optuna-optimized Logistic Regression. Params: {study_lr.best_params}',
            stage='Staging'
        )
        registry_names.append('churn_predictor_lr')
    
    # Decision Tree
    param_dist_dt = {
        'max_depth': lambda trial: trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': lambda trial: trial.suggest_int('min_samples_leaf', 1, 10),
        'criterion': lambda trial: trial.suggest_categorical('criterion', ['gini', 'entropy'])
    }
    
    with mlflow.start_run(run_name='decision_tree_optuna'):
        trainer._log_git_info()
        mlflow.log_param('n_trials', 20)
        
        study_dt, best_dt = trainer.optuna_optimization_with_model(
            DecisionTreeClassifier, param_dist_dt, n_trials=20, model_name='dt_optuna'
        )
        
        dt_metrics = trainer.evaluate_model('dt_optuna', model=best_dt)
        trainer._log_metrics_to_mlflow(dt_metrics)
        trainer.trained_models_metrics['decision_tree'] = dt_metrics
        
        trainer.register_model_to_registry(
            model=best_dt,
            model_name='dt_optuna',
            registry_name='churn_predictor_dt',
            metrics=dt_metrics,
            model_description=f'Optuna-optimized Decision Tree. Params: {study_dt.best_params}',
            stage='Staging'
        )
        registry_names.append('churn_predictor_dt')
    
    # ==================== Model Comparison ====================
    print("\n" + "="*60)
    print("Model Performance Comparison")
    print("="*60)
    
    comparison_df = trainer.compare_all_models()
    print("\n", comparison_df.to_string(index=False))
    
    # ==================== Promote Best Model ====================
    print("\n" + "="*60)
    print("Promoting Best Model to Production")
    print("="*60)
    
    trainer.promote_best_model_across_all(registry_names, metric='f1')
    
    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("Check MLflow UI for detailed tracking and model registry")
    print("="*60)


if __name__ == '__main__':
    main()