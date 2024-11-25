# Heart Disease Prediction Model Evaluation Report

## Dataset Statistics
- Total samples: 63959
- Positive sample ratio: 8.74%

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

