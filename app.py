import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

def load_model(model_path):
    """載入模型文件"""
    try:
        with open(model_path, 'rb') as f:
            dv, scaler, model = pickle.load(f)
        return dv, scaler, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def predict_heart_disease(customer, dv, scaler, model):
    """使用模型進行預測"""
    try:
        # 定義特徵順序
        numerical_features = ['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime']
        
        # 處理數值特徵
        numerical_values = np.array([[float(customer[f]) for f in numerical_features]])
        numerical_scaled = scaler.transform(numerical_values)
        
        # 處理所有特徵
        X = dv.transform([customer])
        
        # 預測
        y_pred = model.predict_proba(X)[0, 1]
        return y_pred
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def get_available_models():
    """獲取models目錄下所有的.bin文件"""
    models_dir = Path('.')
    return list(models_dir.glob('*.bin'))

def main():
    st.title('Heart Disease Prediction App')
    
    try:
        # 載入測試集數據
        df_test = pd.read_pickle('df_test.pkl')
        
        # 側邊欄：模型選擇
        st.sidebar.header('Model Selection')
        available_models = get_available_models()
        if not available_models:
            st.error("No model files found in the directory!")
            return
            
        selected_model_path = st.sidebar.selectbox(
            'Choose a model:',
            options=available_models,
            format_func=lambda x: x.name
        )
        
        # 載入選擇的模型
        dv, scaler, model = load_model(selected_model_path)
        if None in (dv, scaler, model):
            st.error("Failed to load model!")
            return
        
        # 主要內容區域
        st.header('Patient Information')
        
        # 測試集數據選擇
        test_index = st.number_input(
            'Select test set index:',
            min_value=0,
            max_value=len(df_test)-1,
            value=0,
            step=1
        )
        
        # 獲取選擇的測試數據
        selected_patient = df_test.iloc[test_index].to_dict()
        
        # 創建兩列佈局
        col1, col2 = st.columns(2)
        
        # 數值特徵
        with col1:
            st.subheader('Numerical Features')
            numerical_features = ['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime']
            numerical_values = {}
            for feature in numerical_features:
                numerical_values[feature] = st.number_input(
                    f'{feature}:',
                    value=float(selected_patient[feature]),
                    format='%f'
                )
        
        # 類別特徵
        with col2:
            st.subheader('Categorical Features')
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
            
            categorical_values = {}
            for feature, options in categorical_features.items():
                current_value = str(selected_patient[feature]).lower()
                # 確保當前值在選項中
                if current_value not in options:
                    current_value = options[0]
                categorical_values[feature] = st.selectbox(
                    f'{feature}:',
                    options=options,
                    index=options.index(current_value)
                )
        
        # 合併所有特徵
        patient_data = {**numerical_values, **categorical_values}
        
        # 預測按鈕
        if st.button('Predict Heart Disease Risk'):
            # 進行預測
            probability = predict_heart_disease(patient_data, dv, scaler, model)
            
            if probability is not None:
                # 顯示預測結果
                st.header('Prediction Result')
                
                # 使用進度條顯示風險概率
                st.progress(probability)
                
                # 顯示具體數值
                st.metric(
                    label="Heart Disease Risk",
                    value=f"{probability:.1%}"
                )
                
                # 風險等級評估
                risk_level = (
                    "High Risk! Please consult a doctor." if probability >= 0.7
                    else "Medium Risk. Regular check-ups recommended." if probability >= 0.3
                    else "Low Risk. Maintain a healthy lifestyle."
                )
                
                st.info(risk_level)
                
                # 顯示重要特徵值
                st.subheader('Key Risk Factors:')
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("BMI", f"{patient_data['bmi']:.1f}")
                with col2:
                    st.metric("Physical Health", f"{patient_data['physicalhealth']:.1f}")
                with col3:
                    st.metric("Mental Health", f"{patient_data['mentalhealth']:.1f}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check if all required files are present and correctly formatted.")

if __name__ == '__main__':
    main()
    
    
#     