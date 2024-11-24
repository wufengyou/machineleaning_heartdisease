# 心臟病預測模型評估報告

## 資料集統計
- 總樣本數: 63959
- 正樣本比例: 8.74%

## 模型性能總覽
```
            model_name  accuracy  precision   recall       f1  prediction_time
     lightgbm_original  0.914383   0.577128 0.077611 0.136822         0.684002
     logistic_original  0.913914   0.541748 0.099785 0.168529         0.024015
      xgboost_original  0.913273   0.521781 0.096388 0.162717         0.336005
decision_tree_original  0.911303   0.458631 0.080293 0.136661         0.029017
random_forest_original  0.901343   0.336966 0.132690 0.190403         4.588012
         xgboost_smote  0.886052   0.316609 0.261803 0.286609         0.216003
        lightgbm_smote  0.880877   0.314953 0.308476 0.311681         0.663003
   random_forest_smote  0.876218   0.263768 0.232117 0.246932         6.878997
   decision_tree_smote  0.791585   0.234527 0.611230 0.338986         0.040016
        logistic_smote  0.746306   0.225390 0.780401 0.349764         0.018000
```

