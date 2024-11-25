import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import time

class ModelEvaluator:
    def __init__(self, models_dir: str = 'models', data_dir: str = 'data'):
        """Initialize model evaluator"""
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models: Dict[str, Tuple[Any, Any, Any]] = {}
        self.results = []
        self.load_test_data()
        self.load_models()
    
    def load_test_data(self):
        """Load and process test data"""
        try:
            df_test = pd.read_pickle(self.data_dir / 'df_test.pkl')
            
            # Check original data structure
            print("\nOriginal data structure:")
            print(df_test.head())
            print("\nData types:")
            print(df_test.dtypes)
            
            # Ensure correct processing of heartdisease column
            if 'heartdisease' in df_test.columns:
                if df_test['heartdisease'].dtype == 'O':  # If string type
                    self.y_test = (df_test['heartdisease'].str.lower() == 'yes').astype(int)
                    self.X_test = df_test.drop('heartdisease', axis=1)
                else:  # If already numeric type
                    self.y_test = df_test['heartdisease'].astype(int)
                    self.X_test = df_test.drop('heartdisease', axis=1)
            
            # Print target variable distribution
            value_counts = pd.Series(self.y_test).value_counts()
            print("\nTarget variable distribution:")
            print(value_counts)
            print(f"Positive sample ratio: {(self.y_test.mean() * 100):.2f}%")
            print(f"Loaded test data: {len(self.X_test)} records")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def load_models(self):
        """Load all models"""
        for model_path in self.models_dir.glob('*.bin'):
            try:
                with open(model_path, 'rb') as f:
                    dv, scaler, model = pickle.load(f)
                    self.models[model_path.stem] = (dv, scaler, model)
                print(f"Successfully loaded model: {model_path.stem}")
            except Exception as e:
                print(f"Error loading model {model_path.stem}: {str(e)}")
        print(f"Loaded {len(self.models)} models in total")
    
    def prepare_features(self, dv, scaler, X) -> np.ndarray:
        """Prepare feature data"""
        try:
            numerical_features = ['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime']
            categorical_features = [col for col in X.columns if col not in numerical_features]
            
            # Process numerical features
            X_num = scaler.transform(X[numerical_features])
            
            # Process categorical features
            cat_data = X[categorical_features].copy()
            for col in categorical_features:
                if cat_data[col].dtype == 'O':  # If string type
                    cat_data[col] = cat_data[col].str.lower().str.replace(' ', '_')
            
            X_cat = dv.transform(cat_data.to_dict(orient='records'))
            
            return np.hstack([X_num, X_cat])
        except Exception as e:
            print(f"Error during feature preparation: {str(e)}")
            raise

    def evaluate_model(self, model_name: str) -> dict:
        """Evaluate single model"""
        try:
            print(f"\nEvaluating model: {model_name}")
            dv, scaler, model = self.models[model_name]
            X_transformed = self.prepare_features(dv, scaler, self.X_test)
            
            # Time the prediction process
            start_time = time.time()
            y_pred = model.predict(X_transformed)
            y_pred_proba = model.predict_proba(X_transformed)[:, 1]
            prediction_time = time.time() - start_time
            
            # Calculate evaluation metrics
            results = {
                'model_name': model_name,
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, zero_division=0),
                'recall': recall_score(self.y_test, y_pred, zero_division=0),
                'f1': f1_score(self.y_test, y_pred, zero_division=0),
                'prediction_time': prediction_time,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'y_pred_proba': y_pred_proba
            }
            
            # Calculate ROC AUC only for binary classification
            if len(np.unique(self.y_test)) == 2:
                results['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
            else:
                results['roc_auc'] = None
                print("Warning: Cannot calculate ROC AUC score as test set has only one class")
            
            # Print current results
            print("\nEvaluation results:")
            for metric, value in results.items():
                if metric not in ['confusion_matrix', 'y_pred_proba']:
                    print(f"{metric}: {value}")
            
            return results
            
        except Exception as e:
            print(f"Error evaluating model {model_name}: {str(e)}")
            raise

    def evaluate_all_models(self):
        """Evaluate all models"""
        for model_name in self.models.keys():
            try:
                results = self.evaluate_model(model_name)
                self.results.append(results)
            except Exception as e:
                print(f"Error evaluating model {model_name}: {str(e)}")

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        if not self.results:
            print("No evaluation results available for plotting confusion matrices")
            return
            
        n_models = len(self.results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.ravel()
        
        for idx, result in enumerate(self.results):
            sns.heatmap(
                result['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=axes[idx]
            )
            axes[idx].set_title(f"{result['model_name']}\nConfusion Matrix")
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide unused subplots
        for idx in range(len(self.results), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png')
        plt.close()
        print("Confusion matrices saved as 'confusion_matrices.png'")
    
    def generate_report(self):
        """Generate evaluation report"""
        if not self.results:
            print("No evaluation results available for generating report")
            return None
            
        # Create performance metrics DataFrame
        metrics_df = pd.DataFrame([
            {k: v for k, v in r.items() if not isinstance(v, (np.ndarray, list))}
            for r in self.results
        ])
        
        # Sort models by accuracy
        metrics_df = metrics_df.sort_values('accuracy', ascending=False)
        
        # Generate report
        report = "# Heart Disease Prediction Model Evaluation Report\n\n"
        
        report += "## Dataset Statistics\n"
        report += f"- Total samples: {len(self.X_test)}\n"
        report += f"- Positive sample ratio: {(self.y_test.mean() * 100):.2f}%\n\n"
        
        report += "## Model Performance Overview\n"
        performance_table = metrics_df[['model_name', 'accuracy', 'precision', 'recall', 'f1', 'prediction_time']].to_string(index=False)
        report += f"```\n{performance_table}\n```\n\n"
        
        # Save report
        with open('model_evaluation_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\nEvaluation report generated!")
        return metrics_df

def main():
    """Main function"""
    try:
        evaluator = ModelEvaluator()
        evaluator.evaluate_all_models()
        evaluator.plot_confusion_matrices()
        results_df = evaluator.generate_report()
        
        if results_df is not None and not results_df.empty:
            print("\nBest model performance metrics:")
            print(results_df.iloc[0])
            
    except Exception as e:
        print(f"Error during program execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()