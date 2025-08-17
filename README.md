# DataSift

A binary classifier efficiency enhancement tool. Optimizes feature selection within datasets for faster training and improved binary classification performance.

### *Algorithm Logic*
DataSift implements an intelligent backward elimination feature selection algorithm designed to optimize model performance through informed feature pruning. The algorithm combines statistical preprocessing with iterative performance monitoring to identify the optimal feature subset.

1. Preprocessing / Setup Phase
- Variance Filtering: Removes features with variance below a specified threshold (default: 0.1) to eliminate near-constant variables
- Data Preparation: Converts all features to numeric format and creates stratified train-test splits
- Label Encoding: Maps categorical labels to binary values using a provided label mapping

2. Baseline Establishment
- Performs stratified k-fold cross-validation (default: 10 folds) on the full feature set
- Calculates three key performance metrics:
  - ROC-AUC: Area under the Receiver Operating Characteristic curve
  - PR-AUC: Area under the Precision-Recall curve
  - F1 Score: Harmonic mean of precision and recall under an optimized threshold
- A composite score is created, adding all 3 of the above, used to monitor peak performance
- Obtains feature importance rankings using the trained base classifier


3. Backward Elimination Process
The algorithm iteratively removes the least important features while monitoring performance:
- Sequential Removal: Features are eliminated one by one, starting with the lowest importance
- Performance Tracking: After each removal, the model is trained on the newly pruned dataset and the algorithm recalculates all three metrics via cross-validation
- Composite Scores are monitored as a performance indicator
- Best Feature Set Tracking: Continuously tracks the feature subset yielding the highest composite score

4. Stopping Criteria
The algorithm employs multiple safeguards to prevent over-pruning:
- Performance Break: Stops if any individual metric drops by more than 1% from baseline
- Early Stopping: Uses patience mechanism to halt when performance does not improve for a specified number of iterations (default: 3 iterations) 

5. Output
Returns the optimal feature list that maximizes model performance while maintaining statistical rigor through cross-validation.

Before model hyperparameter optimization, which can be an extensively long process, users can run this program as such to simultaneously reduce the amount of data each model must process and increase its performance by eliminating "confusing" and "noisy" features.


### *Workflow*
This demonstration is also included in the project files.
```python
from DataSift import DataSift
from xgboost import XGBClassifier  # can be any classifier you want

# these are just to unpack / load the dataset 
from pathlib import Path
import pickle as pkl

def refine_features():
    classifier = XGBClassifier()  # load up base model

    dataframe_path = Path('../database') / 'VARIANT_df.pkl'
    with open(dataframe_path, 'rb') as infile:
        dataframe = pkl.load(infile)  # load up dataframe to be fed to model

    y_label = 'ClinicalSignificance'  # our Y axis

    label_map = {'Benign': 0, 'Pathogenic': 1}  # binary label map

    refinement_module = DataSift(classifier=classifier, 
                                 dataframe=dataframe, 
                                 y_label=y_label, 
                                 label_map=label_map)
    # could take this one step further and just get it to output dataframe[refinement_module.d_sift()]
    return refinement_module.d_sift()

refine_features()
```

NOTE: currently only works for classifiers containing feature_importances_ attribute!
Future improvements would be to apply this to other classifiers such as Linear models, NNs .etc.
