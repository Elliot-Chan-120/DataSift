import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone


# ultimate goal - improve model performance via selecting optimal features
# base XGB - # straightforward stratified kfold cross validation
# preprocesses dataframe for low variance features + sequentially prunes the least important features
# monitors model performance after stratified k-fold cv and...
# returns feature list that brought about the greatest model performance


class DataSift:
    def __init__(self, classifier,
                 dataframe, y_label, label_map,
                 var_threshold=0.1,
                 test_size=0.2,
                 random_state=42,
                 cv_splits=10,
                 patience=3):

        self.classifier = classifier
        self.dataframe = dataframe
        self.y_label = y_label
        self.label_map = label_map

        self.var_threshold = var_threshold
        self.test_size = test_size
        self.random_state = random_state
        self.cv_splits = cv_splits
        self.patience = patience

        # train test data
        self.X_train = None
        self.y_train = None

        # stratified cv
        self.skf_cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)

        # baseline metrics
        self.pr_auc = 0
        self.roc_auc = 0
        self.pathogenic_f1 = 0
        self.base_composite = self.pr_auc + self.roc_auc + self.pathogenic_f1

    def setup(self):
        """
        Get X and y of dataframe as numerical values - ensure dataframe only includes numbers pls
        :return:
        """
        X = self.dataframe.drop(self.y_label, axis=1)
        y = self.dataframe[self.y_label]

        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.loc[:, X.var() > self.var_threshold]

        y = y.map(self.label_map)

        self.X_train, junk1, self.y_train, junk2 = train_test_split(X, y,
                                                            test_size=0.2,
                                                            stratify=y,
                                                            random_state=self.random_state)

        return self.X_train, self.y_train

    def d_sift(self):
        """
        Utilizes backward sequencing to determine optimal feature combination for model performance
        Eliminates the least important ones sequentially
        :return:
        """
        df_1 = self.setup()
        preproc_deletes = len(self.dataframe.columns) - len(df_1[0].columns)
        print(f"[Preprocessing] {preproc_deletes} columns below threshold variance {self.var_threshold} deleted")

        # now we have base evaluation metrics and a feature dataframe with increasing importance values
        base_roc, base_prc, base_f1, FeatImp_df = self.importance_df()
        base_composite = base_roc + base_prc + base_f1
        feature_list = FeatImp_df['Feature'].to_list()

        best_features = []
        best_roc = 0
        best_prc = 0
        best_f1 = 0
        best_composite = best_roc + best_prc + best_f1

        best_idx = 0
        early_stop_counter = 0
        for feature_removal_count in range(len(feature_list)):
            new_feature_list = feature_list[feature_removal_count + 1:]  # we do +1 because we already did the first round as base
            X_t_new, y_t_new = self.X_train[new_feature_list], self.y_train

            # get metrics from new list + generate new composite score
            new_roc, new_prc, new_f1 = self.cross_validation(X_t_new, y_t_new)
            new_composite = new_roc + new_prc + new_f1

            # first need to check if nothing has decreased by more than 1%
            if (new_roc <= base_roc - 0.01) or (new_prc <= base_prc - 0.01) or (new_f1 <= base_f1 - 0.01):
                print(f"[|Performance break encountered|]: returning optimal feature list with the following metrics"
                      f"Composite: {new_composite} | ROC_AUC {new_roc} | PR_AUC {new_prc} | F1 {new_f1}")
                break
            elif new_composite > base_composite:  # check if it is better than the base stats and override them if so
                best_features = new_feature_list
                best_roc = new_roc
                best_prc = new_prc
                best_f1 = new_f1
                best_composite = best_roc + best_prc + best_f1
                best_idx = feature_removal_count
            elif new_composite > best_composite:  # now check if it is better than the best stats and overwrite them
                best_features = new_feature_list
                best_roc = new_roc
                best_prc = new_prc
                best_f1 = new_f1
                best_composite = best_roc + best_prc + best_f1
                best_idx = feature_removal_count
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # status check per iteration
            print(f"[{feature_removal_count}] Composite: {new_composite} | ROC_AUC {new_roc} | PR_AUC {new_prc} | F1 {new_f1} | Best_idx = {best_idx}")

            if best_features and early_stop_counter == self.patience:
                print(f"[|Patience threshold exceeded|]: breaking early with the following stats & returning optimal feature list:"
                      f"[{best_idx}] Composite: {best_composite} | ROC_AUC {best_roc} | PR_AUC {best_prc} | F1 {best_f1}")
                break

        if best_features:
            return best_features
        else:
            return feature_list

    def importance_df(self):
        """
        :return: Feature importance dataframe for feature refinement loop + starting stats
        0: average_roc -> 1: average_prc -> 2: average_f1 -> 3: feature importance df
        """
        return self.cross_validation(self.X_train, self.y_train, importance_flag=True)

    def cross_validation(self, X_train, y_train, importance_flag=False):
        """
        Performs stratified cv + outputs roc & pr AUCs, & f1 scores <- evaluation metrics for model performance
        \nIf importance_flag is on - will output feature importance dataframe, do this once for the setup
        :return: 0: average_roc -> 1: average_prc -> 2: average_f1
        """
        model = clone(self.classifier)
        roc_scores, pr_scores, f1_scores = [], [], []

        # k-fold cross validation
        for train_idx, val_idx in self.skf_cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba, pos_label=1)

            # evaluation metrics
            f1_set = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_f1 = np.max(f1_set)
            f1_scores.append(best_f1)
            roc_scores.append(roc_auc_score(y_val, y_pred_proba))
            pr_scores.append(auc(recall, precision))

        # get overall metrics
        average_roc = np.mean(roc_scores)
        average_prc = np.mean(pr_scores)
        average_f1 = np.mean(f1_scores)

        if not importance_flag:
            return average_roc, average_prc, average_f1
        else:
            feature_importances = model.feature_importances_
            featimp_df = pd.DataFrame({
                "Feature": self.X_train.columns,
                "Importance": feature_importances,
            })
            featimp_df = featimp_df.sort_values(by="Importance", ascending=True)
            return average_roc, average_prc, average_f1, featimp_df
