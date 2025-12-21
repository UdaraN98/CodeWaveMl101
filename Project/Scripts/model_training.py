import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import optuna
import os


class ModelTrainer:
    """Modular class for training and evaluating ML models"""
    
    def __init__(self, data_path, test_size=0.2, random_state=42):
        """Initialize ModelTrainer with dataset"""
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        
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
    
    def optuna_optimization(self, n_trials=10):
        """Optimize hyperparameters using Optuna"""
        def objective(trial):
            penalty = trial.suggest_categorical('penalty', ['l2'])
            C = trial.suggest_float('C', 0.01, 100.0, log=True)
            max_iter = trial.suggest_int('max_iter', 100, 300)
            
            model = LogisticRegression(penalty=penalty, C=C, max_iter=max_iter, random_state=self.random_state)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            recall = recall_score(self.y_test, y_pred)
            return recall
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Train model with best parameters
        best_params = study.best_params
        model = LogisticRegression(
            penalty='l2',
            C=best_params['C'],
            max_iter=best_params['max_iter'],
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        self.models['lr_optuna'] = model
        print(f"Best parameters (Optuna): {best_params}")
        return study, model
    
    def evaluate_model(self, model_name, y_pred=None, model=None):
        """Evaluate model performance"""
        if y_pred is None and model is not None:
            y_pred = model.predict(self.X_test)
        elif y_pred is None:
            y_pred = self.models[model_name].predict(self.X_test)
        
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(self.y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(self.y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(self.y_test, y_pred):.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred)
        }
    
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
    """Main execution function"""
    # Initialize trainer
    data_path = '/Users/udaranilupul/Documents/Freelancing/CodeWave/CodeWaveMl101/Project/Data/customer_churn_dataset_prepared.csv'
    trainer = ModelTrainer(data_path)
    
    # Load and prepare data
    trainer.load_dataset()
    trainer.split_data()
    
    # Train default model
    trainer.train_logistic_regression(random_state=42, max_iter=1000)
    
    # GridSearchCV
    param_grid_lr = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100],
        'max_iter': [100, 200, 300]
    }
    grid_search = trainer.grid_search_logistic_regression(param_grid_lr)
    
    # RandomizedSearchCV
    param_dist_lr = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100],
        'max_iter': [100, 200, 300]
    }
    random_search = trainer.random_search_logistic_regression(param_dist_lr)
    
    # Optuna optimization
    study, optuna_model = trainer.optuna_optimization(n_trials=10)
    
    # Evaluate all models
    trainer.evaluate_model('lr_default')
    trainer.evaluate_model('lr_grid_search')
    trainer.evaluate_model('lr_random_search')
    trainer.evaluate_model('lr_optuna')
    
    # Plot confusion matrices
    trainer.plot_confusion_matrix('lr_default')
    trainer.plot_confusion_matrix('lr_grid_search')
    trainer.plot_confusion_matrix('lr_optuna')


if __name__ == '__main__':
    main()