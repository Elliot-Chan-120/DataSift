# DataSift

A binary classifier efficiency enhancement tool. Optimizes feature selection within datasets for faster training and improved binary classification performance.

### *Workflow*

Before model hyperparameter optimization, which can be an extensively long process, users can run this program as such to not only reduce the amount of data each model must process, but also increase its performance by eliminating "confusing" and "noisy" features.

This demonstration is also included in the project files.
```python
from DataSift import DataSift
from xgboost import XGBClassifier  # can be any classifier you want

# these are just to unpack / load the dataset 
from pathlib import Path
import pickle as pkl

# NOTE: currently only works for classifiers containing feature_importances_ attribute!
# Future improvements would be to apply this to other classifiers such as Linear models, NNs .etc.

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

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg


