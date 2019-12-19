import os

import joblib
import xgboost
import shap
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from preprocess.feature_label_preparer import FeatureLabelPreparer
from run import get_labels_and_raw_feature_selections


class LogisticRegressionFeatureExtractor:
    def __init__(self, model_file, feature_name, X):
        """

        :param model_file: the full path for the file containing xgboost model
        :param feature_name: pandas series containing name for features
        """
        self.model = joblib.load(model_file)
        self.feature_names = feature_name.values
        self.X = X

    def find_most_important_n_features(self, n):
        most_important_feature_indices_traditional = np.argsort(self.model.coef_)[0, :-(n+1):-1]
        # print(self.model.coef_[0, most_important_feature_indices_traditional])
        # print(self.feature_names[most_important_feature_indices_traditional])

        # shap.initjs()
        explainer = shap.LinearExplainer(self.model, self.X)
        shap_values = explainer.shap_values(self.X)

        # feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))[::-n]
        global_shap_values = np.abs(shap_values).mean(0)
        most_important_feature_indices = np.argsort(global_shap_values)[:-(n+1):-1]

        """
        return self.feature_names[most_important_feature_indices_traditional], \
               self.model.coef_[0, most_important_feature_indices_traditional], \
               self.X.columns[most_important_feature_indices], \
               global_shap_values[most_important_feature_indices]
        """
        """
        return self.feature_names[most_important_feature_indices_traditional], \
               self.model.coef_[0, most_important_feature_indices_traditional],
        """

        return self.X.columns[most_important_feature_indices], \
               global_shap_values[most_important_feature_indices]


if __name__ == '__main__':
    #raw = open('/home/herkut/Desktop/ar_detector/configurations/conf.yml')
    raw = open('/run/media/herkut/hdd-1/TB_genomes/ar_detector/configurations/conf.yml')
    Config.initialize_configurations(raw)

    label_file, raw_feature_selections = get_labels_and_raw_feature_selections('dataset-ii')

    feature_selections = {}
    for k, v in raw_feature_selections.items():
        feature_selections['binary' + '_' + k] = v

    for k, v in feature_selections.items():
        print("Feature importance would be extacted for: " + k)
        raw_label_matrix = FeatureLabelPreparer.get_labels_from_file(os.path.join(Config.dataset_directory, label_file))
        raw_feature_matrix = FeatureLabelPreparer.get_feature_matrix_from_files(v)

    lr = LogisticRegressionFeatureExtractor('/run/media/herkut/hdd-1/TB_genomes/truba/ar_detector_results_dataset-ii_20191205/best_models/lr_accuracy_phenotype_binary_snp_09_bcf_nu_indel_00_platypus_all/lr_Pyrazinamide.sav',
                                             raw_feature_matrix.columns,
                                             raw_feature_matrix)

    _, _, important_features, scores = lr.find_most_important_n_features(10)
    print(important_features)
