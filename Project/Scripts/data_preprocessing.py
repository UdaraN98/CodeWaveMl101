import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import joblib


class DataPreprocessor:
    """Modular data preprocessing pipeline for customer churn dataset using scikit-learn."""
    
    def __init__(self, input_dir, output_dir):
        """
        Initialize preprocessor with input/output directories.
        
        Args:
            input_dir (str): Directory containing the input CSV file
            output_dir (str): Directory to save preprocessed data
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = None
        self.preprocessor = None
        self.X = None
        self.y = None
    
    def load_data(self, filename='customer_churn_dataset-training-master.csv'):
        """Load CSV data from input directory."""
        filepath = self.input_dir / filename
        self.dataset = pd.read_csv(filepath)
        print(f"✓ Data loaded: {self.dataset.shape[0]} rows, {self.dataset.shape[1]} columns")
        return self.dataset
    
    def _build_preprocessing_pipeline(self):
        """Build scikit-learn ColumnTransformer pipeline."""
        
        # Define columns for each transformation
        categorical_one_hot = ['Gender', 'Contract Length']
        ordinal_features = ['Subscription Type']
        numeric_features = [col for col in self.dataset.columns 
                           if col not in categorical_one_hot + ordinal_features + ['CustomerID', 'Churn']
                           and self.dataset[col].dtype in ['int64', 'float64']]
        
        # One-hot encoding transformer
        one_hot_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Ordinal encoding transformer
        ordinal_transformer = Pipeline(steps=[
            ('ordinal', OrdinalEncoder(
                categories=[['Basic', 'Standard', 'Premium']],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ))
        ])
        
        # Combine all transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', one_hot_transformer, categorical_one_hot),
                ('ordinal', ordinal_transformer, ordinal_features),
                ('numeric', 'passthrough', numeric_features)
            ],
            remainder='drop'
        )
        
        self.preprocessor = preprocessor
        print("✓ Preprocessing pipeline built")
        return preprocessor
    
    def apply_preprocessing(self):
        """Apply the preprocessing pipeline to the dataset."""
        self._build_preprocessing_pipeline()
        
        # Handle missing values
        initial_rows = len(self.dataset)
        self.dataset.dropna(inplace=True)
        dropped = initial_rows - len(self.dataset)
        if dropped > 0:
            print(f"✓ Removed {dropped} rows with missing values")
        else:
            print("✓ No missing values found")
        
        # Separate features and target
        self.y = self.dataset['Churn']
        X_temp = self.dataset.drop(columns=['Churn', 'CustomerID'], errors='ignore')
        
        # Apply preprocessing
        X_transformed = self.preprocessor.fit_transform(X_temp)
        
        # Get feature names after transformation
        feature_names = self._get_feature_names_after_transform(X_temp)
        self.X = pd.DataFrame(X_transformed, columns=feature_names)
        
        print(f"✓ Preprocessing applied: {self.X.shape[0]} rows, {self.X.shape[1]} features")
        return self.X, self.y
    
    def _get_feature_names_after_transform(self, X_temp):
        """Extract feature names after transformation."""
        feature_names = []
        
        # One-hot encoded features
        onehot_encoder = self.preprocessor.named_transformers_['onehot']['onehot']
        onehot_features = onehot_encoder.get_feature_names_out(['Gender', 'Contract Length'])
        feature_names.extend(onehot_features)
        
        # Ordinal features
        feature_names.append('Subscription Type_Encoded')
        
        # Numeric features
        numeric_features = [col for col in X_temp.columns 
                           if col not in ['Gender', 'Contract Length', 'Subscription Type']
                           and X_temp[col].dtype in ['int64', 'float64']]
        feature_names.extend(numeric_features)
        
        return feature_names
    


    
    def save_preprocessor(self, filename='preprocessor.pkl'):
        """Save the fitted preprocessor pipeline for future use."""
        if self.preprocessor is None:
            print("⚠ No preprocessor to save")
            return
        
        filepath = self.output_dir / filename
        joblib.dump(self.preprocessor, filepath)
        print(f"✓ Preprocessor pipeline saved: {filepath}")
    
    def save_data(self, filename='customer_churn_dataset_prepared.csv'):
        """Save preprocessed dataset to output directory."""
        if self.X is None:
            print("⚠ Apply preprocessing first")
            return
        
        # Combine features and target
        output_data = pd.concat([self.X, self.y.reset_index(drop=True)], axis=1)
        
        filepath = self.output_dir / filename
        output_data.to_csv(filepath, index=False)
        print(f"✓ Preprocessed data saved: {filepath}")
    
    def run_pipeline(self, visualize=True, n_rfe_features=5, save_preprocessor=True):
        """Execute the complete preprocessing pipeline."""
        print("\n" + "="*60)
        print("Starting Data Preprocessing Pipeline (scikit-learn)")
        print("="*60 + "\n")
        
        self.load_data()
        self.apply_preprocessing()
        
        
        if save_preprocessor:
            self.save_preprocessor()
        
        self.save_data()
        
        print("\n" + "="*60)
        print("Preprocessing Complete!")
        print("="*60 + "\n")
        
        return self.X, self.y


if __name__ == "__main__":
    # Define input and output directories
    input_directory = '/Users/udaranilupul/Documents/Freelancing/CodeWave/CodeWaveMl101/Project/Data'
    output_directory = '/Users/udaranilupul/Documents/Freelancing/CodeWave/CodeWaveMl101/Project/Data'
    
    # Initialize and run preprocessor
    preprocessor = DataPreprocessor(input_directory, output_directory)
    X, y = preprocessor.run_pipeline(save_preprocessor=True)