# DataSift

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&duration=2500&pause=5000&color=81A1C1&width=800&height=50&lines=Sift+through+high-dimensional+data%2C+retain+high-performance+data.)](https://git.io/typing-svg)

# A Research Helper Module designed to Optimize Binary Classifiers

A binary classifier efficiency enhancement tool. Optimizes feature selection within datasets for faster training and improved binary classification efficiency.

This is my own personal research helper module, and is more of a concept to help my productivity and performance with GEM. This is meant to address high dimensionality in the bioinformatics field. In my case, it was meant to optimize the model-training pipeline for gene variant screening via extensive bioinformatics-based feature engineering.

[Nov 8 2025] First test run, Model: Nov7, comparing to Nov5 w/ no DataSift integration
- GEM version utilized: HMM augments
- training time decreased by roughly 12 minutes
- feature count decreased from 540+ to 242, over 50%...
- model accuracy peaked at 0.83 actually, didn't expect that with the extensive feature elimination. All previous runs were stuck at 0.82
- Mean Pathogenic False Negatives went from 5527 -> 5373 -> greater clinical risk balance
- ROC and PR were relatively identical at 0.8987 and 8952 compared to 0.8988 and 0.8956
- F1 suffered slightly from 0.8040 -> 0.8036

**DataSift demonstrates a practical and biologically meaningful approach to optimizing high-dimensional biomedical classifiers. Its ability to halve the feature set without loss of diagnostic accuracy highlights robust signal retention and improved clinical suitability.**


### *Algorithm Logic*
DataSift implements an intelligent backward elimination feature selection algorithm designed to optimize model performance through informed feature pruning. The algorithm combines statistical preprocessing with iterative performance monitoring to identify the optimal feature subset.

1. Preprocessing
- Data Preparation: Converts all features to numeric format and creates stratified train-test splits
- Label Encoding: Maps categorical labels to binary values using a provided label mapping

2. Variance Filtering
- Variance Filtering: Removes features with variance below a specified threshold to eliminate near-constant variables
- Optimized Variance Determination: uses a Binary Search-type algorithm to determine the variance threshold that preserves signal quality while removing noisy features

3. Baseline Establishment
- Performs stratified k-fold cross-validation (default: 10 folds) on the full feature set
- Calculates three key performance metrics:
  - ROC-AUC: Area under the Receiver Operating Characteristic curve
  - PR-AUC: Area under the Precision-Recall curve
  - F1 Score: Harmonic mean of precision and recall under an optimized threshold
- A composite score is created, adding all 3 of the above, used to monitor peak performance
- Obtains averaged feature importance rankings using the trained base classifier

4. Feature Importance-based Backward Elimination
- The algorithm iteratively removes the least important features while monitoring performance:
- Sequential Removal: Features are eliminated one by one, starting with the lowest importance
- Performance Tracking: After each removal, the model is trained on the newly pruned dataset and the algorithm recalculates all three metrics via cross-validation
- Composite Scores are monitored as a performance indicator
- Best Feature Set Tracking: Continuously tracks the feature subset yielding the highest composite score

5. Stopping Criteria
- The algorithm employs multiple safeguards to prevent over-pruning:
- Performance Break: Stops if any individual metric drops by more than 1% from baseline
- Early Stopping: Uses patience mechanism to halt when performance does not improve for a specified number of iterations (default: 3 iterations) 

Refined features are saved in a config file that can be accessed using the following:

5. Class SiftControl
- allows the user to access the config file and apply the refined feature settings to the current model prior to hyperparameter optimization


### *Workflow*
```python
from DataSift import DataSift
from xgboost import XGBClassifier  # can be any classifier you want with a feature_importances_ attribute

    def optimized_model(self, df):
        y_label = 'ClinicalSignificance'
        df = df.loc[:, ~df.columns.duplicated()]

        X = df.drop(y_label, axis=1)
        y = df[y_label]

        X = X.apply(pd.to_numeric, errors= 'coerce')

        label_map = {'Benign': 0, 'Pathogenic': 1}

        y = y.map(label_map)

	# [1] Load up DataSift and line up params
        feature_optimizer = DataSift(classifier_name=self.model_name,
                                    classifier=XGBClassifier(),
                                    dataframe=df,
                                    y_label=y_label,
                                    label_map=label_map,
                                    variance_space=[0.0, 0.3],
                                    optimize_variance=True,
                                    max_runs=20)

        feature_optimizer.Data_Sift()

	# [2] Use SiftControl to access your model's config and optimized feature selection
        control = SiftControl()
        control.LoadConfig(self.model_name)
        refined_feature_list = control.LoadSift()

        X = X[refined_feature_list]

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size = 0.2,
                                                            stratify=y,
                                                            random_state=42)

# Hyperparameter Optimization Below
```

```python
Nov5 Model stats
Optimal Hyperparameters: {'n_estimators': 1674, 'max_depth': 10, 'learning_rate': 0.034561112430304776, 'subsample': 0.9212141915845736, 'colsample_bytree': 0.6016405698933265, 'colsample_bylevel': 0.9329109895929816, 'reg_alpha': 0.7001202050122113, 'reg_lambda': 3.1671750288760134, 'gamma': 1.0033930419124446, 'min_child_weight': 9, 'scale_pos_weight': 1.6075244983571118}
Cross Validation Results: Mean ROC AUC: 0.8968, Mean PR AUC: 0.8927
Mean FNs: 5527.40, Mean FPs: 5427.40
ROC AUC: 0.8988
Precision-Recall AUC: 0.8956
Pathogenic F1-Score: 0.8040
Optimal threshold for pathogenic detection: 0.511
Performance with optimal threshold:
              precision    recall  f1-score   support

           0       0.83      0.85      0.84     41774
           1       0.81      0.80      0.80     34814

    accuracy                           0.82     76588
   macro avg       0.82      0.82      0.82     76588
weighted avg       0.82      0.82      0.82     76588
Confusion Matrix:
[[35308  6466]
 [ 7040 27774]]
```

```python
Nov7 Model stats
Optimal Hyperparameters: {'n_estimators': 1674, 'max_depth': 10, 'learning_rate': 0.034561112430304776, 'subsample': 0.9212141915845736, 'colsample_bytree': 0.6016405698933265, 'colsample_bylevel': 0.9329109895929816, 'reg_alpha': 0.7001202050122113, 'reg_lambda': 3.1671750288760134, 'gamma': 1.0033930419124446, 'min_child_weight': 9, 'scale_pos_weight': 1.6075244983571118}
Cross Validation Results: Mean ROC AUC: 0.8970, Mean PR AUC: 0.8926
Mean FNs: 5373.00, Mean FPs: 5633.20
ROC AUC: 0.8987
Precision-Recall AUC: 0.8952
Pathogenic F1-Score: 0.8036
Optimal threshold for pathogenic detection: 0.524
Performance with optimal threshold:
              precision    recall  f1-score   support

           0       0.83      0.85      0.84     41774
           1       0.82      0.79      0.80     34814

    accuracy                           0.83     76588
   macro avg       0.82      0.82      0.82     76588
weighted avg       0.83      0.83      0.82     76588
Confusion Matrix:
[[35651  6123]
 [ 7264 27550]]
```
Result Summary: Despite removing over half of the features, Nov7 retained almost identical ROC/PR performance, with a minor shift toward higher pathogenic precision and lower recall. Statistically, this difference is within noise suggesting strong feature redundancy in Nov5 and excellent feature selection in Nov7. It's also worth noting that 55% of features being removed + nearly identical performance means simpler decision boundaries and improved speed, intepretability and overfit robustness. Combined with a further decrease in False Negative occurrence, Nov7 is far more suited for clinical deployability. 


Future improvements:
- performance history - baseline model with x features vs new model with refined features
- save removed features with importance
