import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import pickle
from tqdm import tqdm
import time

class ModelTrainer:
    def __init__(self):
        self.numerical_features = ['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime']
        self.categorical_features = ['smoking', 'alcoholdrinking', 'stroke', 'diffwalking', 
                                   'sex', 'agecategory', 'race', 'diabetic', 
                                   'physicalactivity', 'genhealth', 'asthma', 
                                   'kidneydisease', 'skincancer']
        self.models = {
            'logistic': (LogisticRegression, {'random_state': 42, 'max_iter': 1000}),
            'decision_tree': (DecisionTreeClassifier, {'random_state': 42, 'max_depth': 10}),
            'random_forest': (RandomForestClassifier, {'random_state': 42, 'n_estimators': 100}),
            'xgboost': (XGBClassifier, {'random_state': 42, 'n_estimators': 100}),
            'lightgbm': (LGBMClassifier, {'random_state': 42, 'n_estimators': 100})
        }
        
    def prepare_data(self, df):
        """準備數據：標準化數值特徵和編碼類別特徵"""
        # 數值特徵處理
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df[self.numerical_features])
        
        # 類別特徵處理
        cat_data = df[self.categorical_features].copy()
        for col in self.categorical_features:
            cat_data[col] = cat_data[col].str.lower().str.replace(' ', '_')
        
        dv = DictVectorizer(sparse=False)
        X_cat = dv.fit_transform(cat_data.to_dict(orient='records'))
        
        # 合併特徵
        X = np.hstack([X_num, X_cat])
        
        return X, scaler, dv
    
    def train_model(self, X, y, model_class, model_params, use_smote=False):
        """訓練單個模型"""
        if use_smote:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
        
        model = model_class(**model_params)
        model.fit(X, y)
        return model
    
    def train_all_models(self, df_train, y_train, df_test):
        """訓練所有模型並保存"""
        # 準備數據
        X, scaler, dv = self.prepare_data(df_train)
        
        # 保存測試集
        df_test.to_pickle('data/df_test.pkl')
        print("Saved test data")
        
        # 使用tqdm創建進度條
        with tqdm(total=len(self.models) * 2) as pbar:
            for model_name, (model_class, params) in self.models.items():
                # 訓練原始模型
                model = self.train_model(X, y_train, model_class, params, use_smote=False)
                self.save_model(model, scaler, dv, f'models/{model_name}_original.bin')
                pbar.update(1)
                
                # 訓練SMOTE模型
                model_smote = self.train_model(X, y_train, model_class, params, use_smote=True)
                self.save_model(model_smote, scaler, dv, f'models/{model_name}_smote.bin')
                pbar.update(1)
                
                time.sleep(0.1)  # 讓進度條更容易看見
    
    def save_model(self, model, scaler, dv, filename):
        """保存模型和預處理器"""
        with open(filename, 'wb') as f_out:
            pickle.dump((dv, scaler, model), f_out)
        print(f"Saved model: {filename}")

def main():
    # 載入數據
    df = pd.read_csv('heart_2020_cleaned.csv')
    
    # 資料預處理
    df.columns = df.columns.str.lower()
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    
    # 將目標變數轉換為數值
    df['heartdisease'] = (df['heartdisease'] == 'yes').astype(int)
    
    # 分割訓練集和測試集
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    # 準備目標變數
    y_train = df_train.pop('heartdisease').values
    df_test_for_save = df_test.copy()
    y_test = df_test.pop('heartdisease').values
    
    # 創建必要的目錄
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # 訓練模型
    trainer = ModelTrainer()
    trainer.train_all_models(df_train, y_train, df_test_for_save)

if __name__ == "__main__":
    main()