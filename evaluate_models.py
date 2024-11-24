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
        """初始化模型評估器"""
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models: Dict[str, Tuple[Any, Any, Any]] = {}
        self.results = []
        self.load_test_data()
        self.load_models()
    
    def load_test_data(self):
        """載入並處理測試數據"""
        try:
            df_test = pd.read_pickle(self.data_dir / 'df_test.pkl')
            
            # 檢查原始數據結構
            print("\n原始數據結構：")
            print(df_test.head())
            print("\n數據類型：")
            print(df_test.dtypes)
            
            # 確保 heartdisease 列的處理正確
            if 'heartdisease' in df_test.columns:
                if df_test['heartdisease'].dtype == 'O':  # 如果是字符串類型
                    self.y_test = (df_test['heartdisease'].str.lower() == 'yes').astype(int)
                    self.X_test = df_test.drop('heartdisease', axis=1)
                else:  # 如果已經是數值類型
                    self.y_test = df_test['heartdisease'].astype(int)
                    self.X_test = df_test.drop('heartdisease', axis=1)
            
            # 打印目標變數的分布
            value_counts = pd.Series(self.y_test).value_counts()
            print("\n目標變數分布：")
            print(value_counts)
            print(f"正樣本比例: {(self.y_test.mean() * 100):.2f}%")
            print(f"載入測試數據: {len(self.X_test)} 筆")
            
        except Exception as e:
            print(f"載入數據時發生錯誤: {str(e)}")
            raise
    
    def load_models(self):
        """載入所有模型"""
        for model_path in self.models_dir.glob('*.bin'):
            try:
                with open(model_path, 'rb') as f:
                    dv, scaler, model = pickle.load(f)
                    self.models[model_path.stem] = (dv, scaler, model)
                print(f"成功載入模型: {model_path.stem}")
            except Exception as e:
                print(f"載入模型 {model_path.stem} 時發生錯誤: {str(e)}")
        print(f"共載入 {len(self.models)} 個模型")
    
    def prepare_features(self, dv, scaler, X) -> np.ndarray:
        """準備特徵數據"""
        try:
            numerical_features = ['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime']
            categorical_features = [col for col in X.columns if col not in numerical_features]
            
            # 數值特徵處理
            X_num = scaler.transform(X[numerical_features])
            
            # 類別特徵處理
            cat_data = X[categorical_features].copy()
            for col in categorical_features:
                if cat_data[col].dtype == 'O':  # 如果是字符串類型
                    cat_data[col] = cat_data[col].str.lower().str.replace(' ', '_')
            
            X_cat = dv.transform(cat_data.to_dict(orient='records'))
            
            return np.hstack([X_num, X_cat])
        except Exception as e:
            print(f"特徵準備時發生錯誤: {str(e)}")
            raise

    def evaluate_model(self, model_name: str) -> dict:
        """評估單個模型"""
        try:
            print(f"\n評估模型: {model_name}")
            dv, scaler, model = self.models[model_name]
            X_transformed = self.prepare_features(dv, scaler, self.X_test)
            
            # 計時預測過程
            start_time = time.time()
            y_pred = model.predict(X_transformed)
            y_pred_proba = model.predict_proba(X_transformed)[:, 1]
            prediction_time = time.time() - start_time
            
            # 計算評估指標
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
            
            # 只在有兩個類別時計算 ROC AUC
            if len(np.unique(self.y_test)) == 2:
                results['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
            else:
                results['roc_auc'] = None
                print("警告：由於測試集只有單一類別，無法計算 ROC AUC 分數")
            
            # 打印當前結果
            print("\n評估結果：")
            for metric, value in results.items():
                if metric not in ['confusion_matrix', 'y_pred_proba']:
                    print(f"{metric}: {value}")
            
            return results
            
        except Exception as e:
            print(f"評估模型 {model_name} 時發生錯誤: {str(e)}")
            raise

    def evaluate_all_models(self):
        """評估所有模型"""
        for model_name in self.models.keys():
            try:
                results = self.evaluate_model(model_name)
                self.results.append(results)
            except Exception as e:
                print(f"評估模型 {model_name} 時發生錯誤: {str(e)}")

    def plot_confusion_matrices(self):
        """繪製所有模型的混淆矩陣"""
        if not self.results:
            print("沒有可用的評估結果來繪製混淆矩陣")
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
        
        # 隱藏未使用的子圖
        for idx in range(len(self.results), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png')
        plt.close()
        print("混淆矩陣圖已保存為 'confusion_matrices.png'")
    
    def generate_report(self):
        """生成評估報告"""
        if not self.results:
            print("沒有可用的評估結果來生成報告")
            return None
            
        # 創建性能指標DataFrame
        metrics_df = pd.DataFrame([
            {k: v for k, v in r.items() if not isinstance(v, (np.ndarray, list))}
            for r in self.results
        ])
        
        # 排序模型按準確率
        metrics_df = metrics_df.sort_values('accuracy', ascending=False)
        
        # 生成報告
        report = "# 心臟病預測模型評估報告\n\n"
        
        report += "## 資料集統計\n"
        report += f"- 總樣本數: {len(self.X_test)}\n"
        report += f"- 正樣本比例: {(self.y_test.mean() * 100):.2f}%\n\n"
        
        report += "## 模型性能總覽\n"
        performance_table = metrics_df[['model_name', 'accuracy', 'precision', 'recall', 'f1', 'prediction_time']].to_string(index=False)
        report += f"```\n{performance_table}\n```\n\n"
        
        # 保存報告
        with open('model_evaluation_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n評估報告已生成！")
        return metrics_df

def main():
    """主函數"""
    try:
        evaluator = ModelEvaluator()
        evaluator.evaluate_all_models()
        evaluator.plot_confusion_matrices()
        results_df = evaluator.generate_report()
        
        if results_df is not None and not results_df.empty:
            print("\n最佳模型性能指標：")
            print(results_df.iloc[0])
            
    except Exception as e:
        print(f"程式執行時發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main()
