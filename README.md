# Heart Disease Prediction Project

This project implements a machine learning system for predicting heart disease risk, comparing various models including Logistic Regression, Random Forest, XGBoost, and LightGBM. The system includes both original and SMOTE-balanced versions of each model, providing a comprehensive analysis of prediction performance.

## Overview

The application provides:
- Multiple machine learning models for heart disease prediction
- Interactive web interface built with Streamlit
- Comprehensive model evaluation and comparison
- Real-time prediction capabilities
- Visualization of model performance metrics

## Live Demo

You can access the deployed application at: [Your Streamlit Cloud URL]

## Project Structure
```
.
├── app1.py                 # Streamlit web application
├── train.py               # Model training script
├── evaluate_models.py     # Model evaluation script
├── requirements.txt       # Project dependencies
├── models/               # Directory for trained models (created after training)
│   └── .gitkeep
└── data/                # Directory for processed data (created during training)
    └── .gitkeep
```

## Features

- **Multiple Model Support:**
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
  - Both original and SMOTE-balanced versions

- **Interactive Web Interface:**
  - Real-time predictions
  - User-friendly input forms
  - Visual result presentation
  - Model performance comparison

- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC AUC
  - Confusion Matrix

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wufengyou/machineleaning_heartdisease.git
cd machineleaning_heartdisease
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the models:
```bash
python train.py
```
This will:
- Process the input data
- Train all model variants
- Save models to the `models/` directory

2. (Optional) Evaluate model performance:
```bash
python evaluate_models.py
```
This will generate:
- Performance metrics for all models
- Confusion matrices
- ROC curves
- Detailed evaluation report

3. Run the web application:
```bash
streamlit run app1.py
```

## Model Performance

The project includes various models with different characteristics:

| Model Type | Accuracy | Precision | Recall | F1 Score |
|------------|----------|----------|---------|-----------|
| Logistic Regression | X% | X% | X% | X% |
| Random Forest | X% | X% | X% | X% |
| XGBoost | X% | X% | X% | X% |
| LightGBM | X% | X% | X% | X% |

## Input Features

The model considers various health indicators:

- BMI (Body Mass Index)
- Physical Health
- Mental Health
- Sleep Time
- Smoking Status
- Alcohol Consumption
- Previous Stroke History
- Physical Activity Level
- And more...

## Development

To contribute to this project:

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Requirements

- Python 3.9+
- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- imbalanced-learn

For a complete list of dependencies, see `requirements.txt`.

## Known Issues

- Large model files are not included in the repository and need to be generated locally
- Some models may take significant time to train on lower-end hardware

## Future Improvements

- [ ] Add model explanability features (SHAP values)
- [ ] Implement more advanced feature engineering
- [ ] Add support for more model types
- [ ] Improve UI/UX with more visualizations
- [ ] Add data preprocessing options
- [ ] Implement model retraining capability

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data source: [Heart Disease Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset)
- Streamlit for the excellent web framework
- The scikit-learn, XGBoost, and LightGBM communities

## Contact

- Your Name
- GitHub: [@wufengyou](https://github.com/wufengyou)
- Project Link: https://github.com/wufengyou/machineleaning_heartdisease

## Citation

If you use this project in your research or work, please cite:

```
@misc{heartdisease2024,
  author = {Your Name},
  title = {Heart Disease Prediction Project},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/wufengyou/machineleaning_heartdisease}
}
```
