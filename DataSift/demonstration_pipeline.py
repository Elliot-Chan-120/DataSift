from pathlib import Path
import pickle as pkl
from xgboost import XGBClassifier  # this is the base classifier we will be using

from DataSift import DataSift

# NOTE: currently only works for classifiers containing feature_importances_ attribute!
# Future improvements would be to apply this to other classifiers such as Linear models, NNs .etc.

def refine_features():
    classifier = XGBClassifier()

    dataframe_path = Path('../database') / 'VARIANT_df.pkl'
    with open(dataframe_path, 'rb') as infile:
        dataframe = pkl.load(infile)

    y_label = 'ClinicalSignificance'  # our Y axis

    label_map = {'Benign': 0, 'Pathogenic': 1}  # binary label map

    refinement_module = DataSift(classifier=classifier, dataframe=dataframe, y_label=y_label, label_map=label_map)
    # could take this one step further and just get it to output dataframe[refinement_module.d_sift()]
    return refinement_module.d_sift()

refine_features()