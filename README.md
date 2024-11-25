# Heart Disease Prediction Project

This project implements a machine learning model to predict the likelihood of heart disease based on various health indicators and lifestyle factors. The model is deployed as a web service using Flask and Docker.

This project is comparing various models including Logistic Regression, Random Forest, XGBoost, and LightGBM. The system includes both original and SMOTE-balanced versions of each model, providing a comprehensive analysis of prediction performance.

## Problem Description

Heart disease is one of the leading causes of death globally. Early prediction of heart disease can help in making proactive healthcare decisions. This project uses a dataset containing various health parameters to predict whether a person is likely to have heart disease.

The model takes into account factors such as:

* BMI
* Smoking status
* Alcohol consumption
* Physical health
* Mental health
* Age
* And various other health indicators

## Dataset

The dataset used in this project is the "Heart Disease 2020 Survey Data" from Kaggle.

You can download the dataset from: [https://www.kaggle.com/datasets/luyezhang/heart-2020-cleaned](https://www.kaggle.com/datasets/luyezhang/heart-2020-cleaned)

After downloading, place the `heart_2020_cleaned.csv` file in your project directory.

## Exploratory Data Analysis (EDA)

In this project, we conducted a thorough Exploratory Data Analysis (EDA) on the [Heart 2020 Dataset](https://www.kaggle.com/datasets/luyezhang/heart-2020-cleaned) from Kaggle. The following steps and methods were employed to gain insights into the dataset:

file : eda.ipynb

1. **Initial Data Inspection and Cleaning**:

   - Inspected the dataset structure and content using `pandas` functions such as `df.info()` and `df.head()`.
   - Standardized column names to lowercase and replaced spaces with underscores for consistency using `str.lower()` and `str.replace()`.
   - Normalized categorical values (e.g., converting "Yes/No" responses to lowercase) to simplify processing.
2. **Missing Value Analysis**:

   - Used `df.isnull().sum()` to check for missing values. Necessary imputation or removal strategies were applied based on the results.
3. **Target Variable Distribution**:

   - Encoded the target variable, `heartdisease`, as binary values (`1` for heart disease, `0` for no heart disease).
   - Analyzed the class distribution using `value_counts()` and observed an imbalance with fewer positive cases.
4. **Summary Statistics for Features**:

   - Examined statistical summaries of numerical features using `df.describe()` to identify ranges, means, and variances.
   - Checked the uniqueness and distributions of categorical features using `.nunique()` and `.value_counts()`.
5. **Feature Correlation Analysis**:

   - Calculated Pearson correlation coefficients between numerical features and the target variable using `corrwith`.
   - Applied `mutual_info_score` to measure the mutual information between categorical features and the target variable, identifying key predictors.
6. **Data Visualization**:

   - Plotted distributions of categorical variables and their risk proportions (e.g., mean heart disease rates by category).
   - Analyzed the differences (`diff`) and risk ratios between categories to highlight influential features.
7. **Risk Analysis**:

   - Computed risk ratios (RR) for categorical features, evaluating their relative contribution to heart disease risk.
   - Assessed how deviations from the global mean of the target variable indicate feature importance.
8. **Dataset Splitting**:

   - Divided the dataset into training, validation, and testing subsets using Scikit-Learn's `train_test_split` to support model development and evaluation.

These EDA steps provided valuable insights into the data's structure and feature importance, forming a solid foundation for feature engineering and predictive modeling. Detailed analyses and results are documented in the project's data analysis scripts and reports.

## SMOTE Documentation

See detailed documentation in [smote-explanation.md](/smote-explanation.md)

a method for addressing dataset imbalance problems, commonly used in classification tasks to balance the sample sizes between majority and minority classes

## Overview

The application provides:

- Multiple machine learning models for heart disease prediction
- Interactive web interface built with Streamlit
- Comprehensive model evaluation and comparison
- Real-time prediction capabilities
- Visualization of model performance metrics

## Live Demo

You can access the deployed application at: [Your Streamlit Cloud URL]

## Containerization

```bash
docker build -t heart-disease-predictor .
docker run -p 8501:8501 -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data heart-disease-predictor
```

notes : --volume   can get model and testdata from here : 

[model file](https://www.dropbox.com/scl/fi/duxvgv34csb38hl0dh3iy/model_testdata.zip?rlkey=nzui38fi7ohvctog18b2bibco&st=ltspv0j5&dl=0)  


## Project Structure

```
.
├── app1.py                 # Streamlit web application
├── train.py               # Model training script
├── evaluate_models.py     # Model evaluation script
├── requirements.txt       # Project dependencies
├── models/               # Directory for trained models (created after training)
│   └── .gitkeep            # models go here 
└── data/                # Directory for processed data (created during training)
    └── .gitkeep            #  testdata go here
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

## Installation (optional ,unless you want to train your own model )

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
- or simply download the trained model and test data from following url
- https://www.dropbox.com/scl/fi/duxvgv34csb38hl0dh3iy/model_testdata.zip?rlkey=nzui38fi7ohvctog18b2bibco&st=ltspv0j5&dl=0

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

Dataset Statistics

- Total samples: 63959
- Positive sample ratio: 8.74%

![model confusing matrics](https://github.com/wufengyou/machineleaning_heartdisease/blob/main/confusion_matrices.png)

## Model Performance Overview

```
            model_name  accuracy  precision   recall       f1  prediction_time
     lightgbm_original  0.914383   0.577128 0.077611 0.136822         0.491285
     logistic_original  0.913914   0.541748 0.099785 0.168529         0.020292
      xgboost_original  0.913273   0.521781 0.096388 0.162717         0.122442
decision_tree_original  0.911303   0.458631 0.080293 0.136661         0.040345
random_forest_original  0.901343   0.336966 0.132690 0.190403         4.018701
         xgboost_smote  0.886052   0.316609 0.261803 0.286609         0.120643
        lightgbm_smote  0.880877   0.314953 0.308476 0.311681         0.331418
   random_forest_smote  0.876218   0.263768 0.232117 0.246932         4.941986
   decision_tree_smote  0.791585   0.234527 0.611230 0.338986         0.034569
        logistic_smote  0.746306   0.225390 0.780401 0.349764         0.020484
```

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



## Cloud deployment

This app was deployed in streamlit cloud  as well :

[Streamlit](https://machineleaningheartdisease-2strpccmzvbjeboadwzxxz.streamlit.app/)


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

```

```
