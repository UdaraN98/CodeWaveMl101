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
import shutil
import time
import pickle
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
        self.trained_models_metrics = {}
        self.model_run_ids = {}  # Track run IDs for each model
        
        # MLflow setup
        self.mlflow_tracking_uri = mlflow_tracking_uri or "./mlruns"
        self.experiment_name = experiment_name or "default_experiment"
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                experiment_id = experiment.experiment_id
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            print(f"Warning: Could not set experiment: {e}")
        
        self.mlflow_client = MlflowClient(tracking_uri=self.mlflow_tracking_uri)
        
        # Production artifacts directory
        self.prod_artifacts_dir = os.path.join('mlruns', 'production_models')
        os.makedirs(self.prod_artifacts_dir, exist_ok=True)
        
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
        """Log metrics to MLflow"""
        for metric_name, metric_value in metrics.to_dict().items():
            mlflow.log_metric(metric_name, metric_value)
    
    def _save_production_model(self, model, model_name: str, registry_name: str, 
                               version: str, metrics: ModelMetrics):
        """Save production model directly to file system for API access"""
        try:
            # Create production model directory
            prod_model_dir = os.path.join(self.prod_artifacts_dir, f'{registry_name}_v{version}')
            os.makedirs(prod_model_dir, exist_ok=True)
            
            # Save model using pickle
            model_path = os.path.join(prod_model_dir, 'model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'registry_name': registry_name,
                'version': version,
                'promoted_at': datetime.now().isoformat(),
                'model_path': model_path,
                'metrics': metrics.to_dict()
            }
            
            metadata_path = os.path.join(prod_model_dir, 'production_metadata.txt')
            with open(metadata_path, 'w') as f:
                f.write(f"Production Model Metadata\n")
                f.write(f"{'='*50}\n")
                for key, value in metadata.items():
                    if key == 'metrics':
                        f.write(f"\nMetrics:\n")
                        for metric_name, metric_value in value.items():
                            f.write(f"  {metric_name}: {metric_value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                
                if self.git_info:
                    f.write(f"\nGit Information:\n")
                    f.write(f"  commit: {self.git_info.commit_hash}\n")
                    f.write(f"  branch: {self.git_info.branch}\n")
                    f.write(f"  author: {self.git_info.author}\n")
            
            # Save a simple load script
            load_script = f"""# Load Production Model
import pickle

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Use the model
# predictions = model.predict(X_new)
"""
            
            with open(os.path.join(prod_model_dir, 'load_model.py'), 'w') as f:
                f.write(load_script)
            
            print(f"  ✓ Production model saved to: {prod_model_dir}")
            print(f"  → Model ready for API at: {model_path}")
            
            return prod_model_dir
            
        except Exception as e:
            print(f"  ✗ Error saving production model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def register_model_to_registry(self, model, model_name: str, registry_name: str,
                                   metrics: ModelMetrics, run_id: str = None,
                                   model_description: str = None,
                                   stage: str = 'Staging') -> str:
        """Register a model to MLflow Model Registry"""
        try:
            # If we have a run_id, use it to register the model
            if run_id:
                # Get the artifact URI from the run
                run = self.mlflow_client.get_run(run_id)
                model_uri = f"runs:/{run_id}/{model_name}"
            else:
                # Log the model in the current run
                signature = infer_signature(self.X_train, model.predict(self.X_train))
                model_info = mlflow.sklearn.log_model(
                    model, 
                    artifact_path=model_name,
                    signature=signature
                )
                model_uri = model_info.model_uri
                run_id = mlflow.active_run().info.run_id
            
            # Register the model
            try:
                registered_model = mlflow.register_model(model_uri, registry_name)
                model_version = registered_model.version
                print(f"✓ Model registered: {registry_name} version {model_version}")
            except Exception as reg_error:
                print(f"Registration error: {reg_error}")
                # Try to get the latest version
                try:
                    versions = self.mlflow_client.search_model_versions(f"name='{registry_name}'")
                    if versions:
                        model_version = max([int(v.version) for v in versions])
                    else:
                        print(f"Could not determine version for {registry_name}")
                        return None
                except Exception as e:
                    print(f"Error getting model version: {e}")
                    return None
            
            # Wait a bit for the registry to update
            time.sleep(2)
            
            # Set tags
            tags = {
                'registered_at': datetime.now().isoformat(),
                'f1_score': str(round(metrics.f1, 4)),
                'accuracy': str(round(metrics.accuracy, 4)),
                'auc': str(round(metrics.auc, 4)),
                'run_id': run_id
            }
            
            if self.git_info:
                tags['git_commit'] = self.git_info.commit_hash[:8]
                tags['git_author'] = self.git_info.author
            
            # Set tags with retry
            for attempt in range(3):
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
                    if attempt < 2:
                        time.sleep(1)
                    else:
                        print(f"Warning: Could not set all tags: {e}")
            
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
                print(f"✓ Model stage set to: {stage}")
            except Exception as e:
                print(f"Warning: Could not set stage: {e}")
            
            print(f"  F1: {metrics.f1:.4f} | Accuracy: {metrics.accuracy:.4f} | AUC: {metrics.auc:.4f}")
            
            return str(model_version)
            
        except Exception as e:
            print(f"Error in register_model_to_registry: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
            params = {name: suggest_func(trial) for name, suggest_func in param_distributions.items()}
            model = model_class(**params)
            model.fit(self.X_train, self.y_train)
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
    
    def save_best_model_to_production(self) -> Dict[str, Any]:
        """Find the best model and save it to production"""
        if not self.trained_models_metrics:
            print("No models have been trained yet!")
            return None
        
        # Find best model by F1 score
        best_model_name = max(self.trained_models_metrics.items(), 
                             key=lambda x: x[1].f1)[0]
        best_metrics = self.trained_models_metrics[best_model_name]
        best_model = self.models[best_model_name]
        
        print(f"\n{'='*60}")
        print("SAVING BEST MODEL TO PRODUCTION")
        print(f"{'='*60}")
        print(f"Best Model: {best_model_name}")
        print(f"F1 Score: {best_metrics.f1:.4f}")
        print(f"Accuracy: {best_metrics.accuracy:.4f}")
        print(f"AUC: {best_metrics.auc:.4f}")
        print(f"{'='*60}\n")
        
        # Save to production folder
        registry_name = f"churn_predictor_{best_model_name}"
        version = "1"  # You can increment this if needed
        
        prod_dir = self._save_production_model(
            model=best_model,
            model_name=best_model_name,
            registry_name=registry_name,
            version=version,
            metrics=best_metrics
        )
        
        return {
            'model_name': best_model_name,
            'registry_name': registry_name,
            'version': version,
            'metrics': best_metrics.to_dict(),
            'production_dir': prod_dir
        }


def main():
    """Main execution function with optimized workflow"""
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Preprocessing
    print("="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    preprocessor = DataPreprocessor(input_dir=data_dir, output_dir=data_dir)
    preprocessor.load_data()
    preprocessor.apply_preprocessing()
    preprocessor.save_data()
    
    # Initialize trainer
    data_path = os.path.join(data_dir, 'customer_churn_dataset_prepared.csv')
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', './mlruns')
    
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
    
    # ==================== Train and Optimize Models ====================
    print("\n" + "="*60)
    print("TRAINING MODELS WITH OPTUNA OPTIMIZATION")
    print("="*60)
    
    # Logistic Regression
    print("\n[1/2] Training Logistic Regression...")
    param_dist_lr = {
        'C': lambda trial: trial.suggest_float('C', 0.001, 100.0, log=True),
        'max_iter': lambda trial: trial.suggest_int('max_iter', 100, 500),
        'solver': lambda trial: trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg'])
    }
    
    with mlflow.start_run(run_name='logistic_regression_optuna') as run:
        lr_run_id = run.info.run_id
        trainer._log_git_info()
        mlflow.log_param('n_trials', 20)
        mlflow.log_param('model_type', 'logistic_regression')
        
        study_lr, best_lr = trainer.optuna_optimization_with_model(
            LogisticRegression, param_dist_lr, n_trials=20, model_name='lr_optuna'
        )
        
        # Log best params
        for param_name, param_value in study_lr.best_params.items():
            mlflow.log_param(f'best_{param_name}', param_value)
        
        lr_metrics = trainer.evaluate_model('lr_optuna', model=best_lr)
        trainer._log_metrics_to_mlflow(lr_metrics)
        trainer.trained_models_metrics['logistic_regression'] = lr_metrics
        trainer.model_run_ids['logistic_regression'] = lr_run_id
        
        # Register to MLflow
        version = trainer.register_model_to_registry(
            model=best_lr,
            model_name='lr_optuna',
            registry_name='churn_predictor_lr',
            metrics=lr_metrics,
            run_id=lr_run_id,
            model_description=f'Optuna-optimized Logistic Regression. Best params: {study_lr.best_params}',
            stage='Staging'
        )
    
    # Decision Tree
    print("\n[2/2] Training Decision Tree...")
    param_dist_dt = {
        'max_depth': lambda trial: trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': lambda trial: trial.suggest_int('min_samples_leaf', 1, 10),
        'criterion': lambda trial: trial.suggest_categorical('criterion', ['gini', 'entropy'])
    }
    
    with mlflow.start_run(run_name='decision_tree_optuna') as run:
        dt_run_id = run.info.run_id
        trainer._log_git_info()
        mlflow.log_param('n_trials', 20)
        mlflow.log_param('model_type', 'decision_tree')
        
        study_dt, best_dt = trainer.optuna_optimization_with_model(
            DecisionTreeClassifier, param_dist_dt, n_trials=20, model_name='dt_optuna'
        )
        
        # Log best params
        for param_name, param_value in study_dt.best_params.items():
            mlflow.log_param(f'best_{param_name}', param_value)
        
        dt_metrics = trainer.evaluate_model('dt_optuna', model=best_dt)
        trainer._log_metrics_to_mlflow(dt_metrics)
        trainer.trained_models_metrics['decision_tree'] = dt_metrics
        trainer.model_run_ids['decision_tree'] = dt_run_id
        
        # Register to MLflow
        version = trainer.register_model_to_registry(
            model=best_dt,
            model_name='dt_optuna',
            registry_name='churn_predictor_dt',
            metrics=dt_metrics,
            run_id=dt_run_id,
            model_description=f'Optuna-optimized Decision Tree. Best params: {study_dt.best_params}',
            stage='Staging'
        )
    
    # ==================== Model Comparison ====================
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    comparison_df = trainer.compare_all_models()
    print("\n", comparison_df.to_string(index=False))
    
    # ==================== Save Best Model to Production ====================
    best_model_info = trainer.save_best_model_to_production()
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"Best Model: {best_model_info['model_name']}")
    print(f"Production Directory: {best_model_info['production_dir']}")
    print(f"\nTo use the model in your API:")
    print(f"  import pickle")
    print(f"  with open('{best_model_info['production_dir']}/model.pkl', 'rb') as f:")
    print(f"      model = pickle.load(f)")
    print("="*60)


if __name__ == '__main__':
    main()