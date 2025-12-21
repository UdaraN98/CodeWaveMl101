import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import joblib
from pathlib import Path
from datetime import datetime
import subprocess
import json

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from model_training import ModelTrainer


class GitTracker:
    """Track Git information for experiments"""
    
    @staticmethod
    def get_commit_hash():
        """Get current Git commit hash"""
        try:
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.STDOUT
            ).decode('utf-8').strip()
            return commit_hash
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"
    
    @staticmethod
    def get_branch_name():
        """Get current Git branch name"""
        try:
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.STDOUT
            ).decode('utf-8').strip()
            return branch
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"
    
    @staticmethod
    def get_git_status():
        """Get Git status (dirty/clean)"""
        try:
            status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.STDOUT
            ).decode('utf-8').strip()
            return "clean" if not status else "dirty"
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"
    
    @staticmethod
    def get_commit_message():
        """Get latest commit message"""
        try:
            message = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=%B'],
                stderr=subprocess.STDOUT
            ).decode('utf-8').strip()
            return message
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"
    
    @staticmethod
    def log_git_info():
        """Return dictionary with all Git information"""
        return {
            'commit_hash': GitTracker.get_commit_hash(),
            'branch_name': GitTracker.get_branch_name(),
            'git_status': GitTracker.get_git_status(),
            'commit_message': GitTracker.get_commit_message()
        }


class ExperimentTracker:
    """Track ML experiments using MLflow with Optuna hyperparameter tuning"""
    
    def __init__(self, data_path, mlflow_tracking_uri=None, experiment_name='Customer_Churn_testing'):
        """
        Initialize ExperimentTracker
        
        Args:
            data_path (str): Path to preprocessed dataset
            mlflow_tracking_uri (str): MLflow tracking server URI (default: local)
            experiment_name (str): Name of the MLflow experiment
        """
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.models_registry = {}
        self.best_models = {}
        self.best_params = {}
        self.best_scores = {}
        
        # Get Git information
        self.git_info = GitTracker.log_git_info()
        
        # Setup MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        
        # Create model registry directory
        self.registry_dir = Path('./model_registry')
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.trainer = ModelTrainer(data_path)
        self.trainer.load_dataset()
        self.trainer.split_data()
        
        print(f"✓ ExperimentTracker initialized")
        print(f"  - Experiment: {experiment_name}")
        print(f"  - Train set: {self.trainer.X_train.shape}")
        print(f"  - Test set: {self.trainer.X_test.shape}")
        print(f"  - Git Commit: {self.git_info['commit_hash'][:8]}")
        print(f"  - Git Branch: {self.git_info['branch_name']}")
        print(f"  - Git Status: {self.git_info['git_status']}")
    
    def objective_logistic_regression(self, trial):
        """Optuna objective function for Logistic Regression"""
        params = {
            'C': trial.suggest_float('C', 0.001, 100, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'max_iter': trial.suggest_int('max_iter', 100, 500),
            'solver': 'liblinear',
            'random_state': 42
        }
        
        model = LogisticRegression(**params)
        model.fit(self.trainer.X_train, self.trainer.y_train)
        y_pred = model.predict(self.trainer.X_test)
        
        # Use F1 as optimization metric
        score = f1_score(self.trainer.y_test, y_pred)
        return score
    
    def objective_svm(self, trial):
        """Optuna objective function for SVM"""
        params = {
            'C': trial.suggest_float('C', 0.1, 100, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'max_iter': trial.suggest_int('max_iter', 100, 500),
            'random_state': 42
        }
        
        model = SVC(**params)
        model.fit(self.trainer.X_train, self.trainer.y_train)
        y_pred = model.predict(self.trainer.X_test)
        
        score = f1_score(self.trainer.y_test, y_pred)
        return score
    
    def objective_knn(self, trial):
        """Optuna objective function for KNN"""
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan']),
            'p': trial.suggest_int('p', 1, 2)
        }
        
        model = KNeighborsClassifier(**params)
        model.fit(self.trainer.X_train, self.trainer.y_train)
        y_pred = model.predict(self.trainer.X_test)
        
        score = f1_score(self.trainer.y_test, y_pred)
        return score
    
    def objective_decision_tree(self, trial):
        """Optuna objective function for Decision Tree"""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
            'random_state': 42
        }
        
        model = DecisionTreeClassifier(**params)
        model.fit(self.trainer.X_train, self.trainer.y_train)
        y_pred = model.predict(self.trainer.X_test)
        
        score = f1_score(self.trainer.y_test, y_pred)
        return score
    
    def optimize_model(self, model_type, n_trials=20, timeout = 600):
        """
        Optimize hyperparameters using Optuna for a specific model
        
        Args:
            model_type (str): 'logistic_regression', 'svm', 'knn', or 'decision_tree'
            n_trials (int): Number of optimization trials
            timeout (int): Timeout in seconds
        """
        print(f"\n{'='*60}")
        print(f"Starting Optuna Optimization for {model_type.upper()}")
        print(f"{'='*60}")
        
        # Select objective function
        objectives = {
            'logistic_regression': self.objective_logistic_regression,
            'svm': self.objective_svm,
            'knn': self.objective_knn,
            'decision_tree': self.objective_decision_tree
        }
        
        if model_type not in objectives:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner()
        )
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_type}_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log Git information
            mlflow.log_param('git_commit_hash', self.git_info['commit_hash'])
            mlflow.log_param('git_branch', self.git_info['branch_name'])
            mlflow.log_param('git_status', self.git_info['git_status'])
            
            # Log experiment parameters
            mlflow.log_param('model_type', model_type)
            mlflow.log_param('n_trials', n_trials)
            mlflow.log_param('timeout', timeout)
            mlflow.log_param('optimization_metric', 'f1_score')
            
            # Log commit message as artifact
            self._log_git_info_artifact()
            
            # Run optimization
            study.optimize(
                objectives[model_type],
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True
            )
            
            # Get best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            self.best_params[model_type] = best_params
            self.best_scores[model_type] = best_value
            
            print(f"\n✓ Best F1 Score: {best_value:.4f}")
            print(f"✓ Best Parameters: {best_params}")
            
            # Log best parameters
            for param_name, param_value in best_params.items():
                mlflow.log_param(f'best_{param_name}', param_value)
            
            mlflow.log_metric('best_f1_score', best_value)
            
            # Train final model with best parameters
            final_model = self._train_final_model(model_type, best_params)
            self.best_models[model_type] = final_model
            
            # Evaluate final model
            metrics = self._evaluate_and_log(model_type, final_model)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                final_model,
                artifact_path="model",
                registered_model_name=f"{model_type}_best_model"
            )
            
            # Save model locally
            self._save_model(model_type, final_model, best_params)
            
        return study, final_model
    
    def _log_git_info_artifact(self):
        """Log Git information as a JSON artifact"""
        git_info_path = Path('./git_info.json')
        with open(git_info_path, 'w') as f:
            json.dump(self.git_info, f, indent=2)
        
        mlflow.log_artifact(str(git_info_path))
        git_info_path.unlink()  # Clean up temporary file
    
    def _train_final_model(self, model_type, params):
        """Train final model with best parameters"""
        models = {
            'logistic_regression': LogisticRegression(**params),
            'svm': SVC(**params),
            'knn': KNeighborsClassifier(**params),
            'decision_tree': DecisionTreeClassifier(**params)
        }
        
        model = models[model_type]
        model.fit(self.trainer.X_train, self.trainer.y_train)
        return model
    
    def _evaluate_and_log(self, model_type, model):
        """Evaluate model and log metrics to MLflow"""
        y_pred = model.predict(self.trainer.X_test)
        
        metrics = {
            'accuracy': accuracy_score(self.trainer.y_test, y_pred),
            'precision': precision_score(self.trainer.y_test, y_pred),
            'recall': recall_score(self.trainer.y_test, y_pred),
            'f1': f1_score(self.trainer.y_test, y_pred)
        }
        
        print(f"\n{'='*50}")
        print(f"Final Model Evaluation - {model_type.upper()}")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        return metrics
    
    def _save_model(self, model_type, model, params):
        """Save model and parameters to registry"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        commit_short = self.git_info['commit_hash'][:8]
        
        model_path = self.registry_dir / f"{model_type}_{timestamp}_{commit_short}.pkl"
        params_path = self.registry_dir / f"{model_type}_{timestamp}_{commit_short}_params.txt"
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save parameters with Git info
        with open(params_path, 'w') as f:
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Git Commit: {self.git_info['commit_hash']}\n")
            f.write(f"Git Branch: {self.git_info['branch_name']}\n")
            f.write(f"Git Status: {self.git_info['git_status']}\n")
            f.write(f"Commit Message: {self.git_info['commit_message']}\n")
            f.write(f"\nParameters:\n")
            for param_name, param_value in params.items():
                f.write(f"  {param_name}: {param_value}\n")
        
        print(f"✓ Model saved: {model_path}")
        print(f"✓ Params saved: {params_path}")
    
    def run_all_experiments(self, n_trials=20):
        """Run hyperparameter tuning for all models"""
        models_to_tune = ['logistic_regression', 'svm', 'knn', 'decision_tree']
        
        print(f"\n{'='*70}")
        print(f"STARTING ALL MODEL EXPERIMENTS")
        print(f"Commit Hash: {self.git_info['commit_hash'][:8]}")
        print(f"Branch: {self.git_info['branch_name']}")
        print(f"{'='*70}")
        
        for model_type in models_to_tune:
            try:
                self.optimize_model(model_type, n_trials=n_trials)
            except Exception as e:
                print(f"✗ Error optimizing {model_type}: {str(e)}")
        
        # Compare all models
        self._compare_all_models()
    
    def _compare_all_models(self):
        """Compare performance of all models"""
        print(f"\n{'='*70}")
        print(f"MODEL COMPARISON SUMMARY")
        print(f"{'='*70}\n")
        
        comparison_df = pd.DataFrame({
            'Model': list(self.best_scores.keys()),
            'Best_F1_Score': list(self.best_scores.values())
        }).sort_values('Best_F1_Score', ascending=False)
        
        print(comparison_df.to_string(index=False))
        
        best_model = comparison_df.iloc[0]['Model']
        best_score = comparison_df.iloc[0]['Best_F1_Score']
        
        print(f"\n✓ Best Overall Model: {best_model} (F1: {best_score:.4f})")
        
        # Log to MLflow
        with mlflow.start_run(run_name="model_comparison"):
            mlflow.log_param('git_commit_hash', self.git_info['commit_hash'])
            mlflow.log_table(comparison_df, artifact_file="comparison.json")
            mlflow.log_metric("best_model_f1", best_score)
    
    def get_best_model(self, model_type):
        """Retrieve best model for a specific model type"""
        if model_type in self.best_models:
            return self.best_models[model_type]
        else:
            raise ValueError(f"No optimized model found for {model_type}")


def main():
    """Main execution function"""
    # Initialize tracker
    data_path = '/Users/udaranilupul/Documents/Freelancing/CodeWave/CodeWaveMl101/Project/Data/customer_churn_dataset_prepared.csv'
    
    tracker = ExperimentTracker(
        data_path=data_path,
        experiment_name='Customer_Churn_HPO_Experiment_2'
    )
    
    # Run all experiments
    tracker.run_all_experiments(n_trials=10)
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENTS COMPLETED!")
    print(f"Check MLflow UI with: mlflow ui")
    print(f"Model registry location: {tracker.registry_dir}")
    print(f"Commit Hash: {tracker.git_info['commit_hash']}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()