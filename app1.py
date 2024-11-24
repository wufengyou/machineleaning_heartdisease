import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import os

class HeartDiseasePredictor:
    def __init__(self):
        self.load_data()
        self.setup_page()
    
    def load_data(self):
        """載入測試數據和獲取可用的模型"""
        self.df_test = pd.read_pickle('data/df_test.pkl')
        self.models_path = Path('models')
        self.available_models = list(self.models_path.glob('*.bin'))
    
    def setup_page(self):
        """設置頁面佈局"""
        st.title('Heart Disease Prediction App')
        st.sidebar.title('Model Selection')
        
        # 模型選擇
        self.selected_model_path = st.sidebar.selectbox(
            'Choose a model:',
            options=self.available_models,
            format_func=lambda x: x.stem  # 只顯示文件名，不顯示.bin
        )
        
        # 顯示模型描述
        model_type = self.selected_model_path.stem.split('_')[0]
        is_smote = 'smote' in self.selected_model_path.stem
        
        st.sidebar.markdown(f"""
        **Selected Model Info:**
        - Type: {model_type.upper()}
        - SMOTE: {'Yes' if is_smote else 'No'}
        """)
    
    def load_model(self, model_path):
        """載入選擇的模型"""
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def create_feature_input(self):
        """創建特徵輸入界面"""
        st.header('Patient Information')
        
        # 選擇測試集樣本
        test_index = st.number_input(
            'Select test set index:',
            min_value=0,
            max_value=len(self.df_test)-1,
            value=0
        )
        
        selected_patient = self.df_test.iloc[test_index].to_dict()
        
        col1, col2 = st.columns(2)
        
        # 數值特徵
        numerical_features = ['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime']
        with col1:
            st.subheader('Numerical Features')
            numerical_values = {}
            for feature in numerical_features:
                numerical_values[feature] = st.number_input(
                    f'{feature}:',
                    value=float(selected_patient[feature]),
                    format='%f'
                )
        
        # 類別特徵和選項
        categorical_features = {
            'smoking': ['yes', 'no'],
            'alcoholdrinking': ['yes', 'no'],
            'stroke': ['yes', 'no'],
            'diffwalking': ['yes', 'no'],
            'sex': ['female', 'male'],
            'agecategory': ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', 
                           '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80_or_older'],
            'race': ['white', 'black', 'asian', 'hispanic', 'american_indian/alaskan_native', 'other'],
            'diabetic': ['yes', 'no', 'no,_borderline_diabetes', 'yes_(during_pregnancy)'],
            'physicalactivity': ['yes', 'no'],
            'genhealth': ['excellent', 'very_good', 'good', 'fair', 'poor'],
            'asthma': ['yes', 'no'],
            'kidneydisease': ['yes', 'no'],
            'skincancer': ['yes', 'no']
        }
        
        with col2:
            st.subheader('Categorical Features')
            categorical_values = {}
            for feature, options in categorical_features.items():
                current_value = str(selected_patient[feature]).lower()
                categorical_values[feature] = st.selectbox(
                    f'{feature}:',
                    options=options,
                    index=options.index(current_value) if current_value in options else 0
                )
        
        return {**numerical_values, **categorical_values}
    
    def predict(self, patient_data, dv, scaler, model):
        """進行預測"""
        try:
            # 將數據轉換為DataFrame
            patient_df = pd.DataFrame([patient_data])
            
            # 處理數值特徵
            numerical_features = ['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime']
            X_num = scaler.transform(patient_df[numerical_features])
            
            # 處理類別特徵
            categorical_dict = {k: v for k, v in patient_data.items() 
                              if k not in numerical_features}
            X_cat = dv.transform([categorical_dict])
            
            # 合併特徵
            X = np.hstack([X_num, X_cat])
            
            # 預測
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)[0, 1]
            else:
                return model.predict(X)[0]
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    def run(self):
        """運行應用"""
        # 載入選擇的模型
        dv, scaler, model = self.load_model(self.selected_model_path)
        
        # 創建特徵輸入界面
        patient_data = self.create_feature_input()
        
        # 預測按鈕
        if st.button('Predict Heart Disease Risk'):
            probability = self.predict(patient_data, dv, scaler, model)
            
            if probability is not None:
                st.header('Prediction Result')
                
                # 進度條顯示風險
                st.progress(probability)
                
                # 顯示具體數值
                st.metric(
                    label="Heart Disease Risk",
                    value=f"{probability:.1%}"
                )
                
                # 風險等級
                risk_level = (
                    "🔴 High Risk! Please consult a doctor." if probability >= 0.7
                    else "🟡 Medium Risk. Regular check-ups recommended." if probability >= 0.3
                    else "🟢 Low Risk. Maintain a healthy lifestyle."
                )
                
                st.info(risk_level)
                
                # 顯示關鍵風險因素
                st.subheader('Key Risk Factors')
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("BMI", f"{patient_data['bmi']:.1f}")
                with col2:
                    st.metric("Physical Health", f"{patient_data['physicalhealth']:.1f}")
                with col3:
                    st.metric("Mental Health", f"{patient_data['mentalhealth']:.1f}")

if __name__ == "__main__":
    app = HeartDiseasePredictor()
    app.run()