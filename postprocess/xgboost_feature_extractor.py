import os

import joblib
import xgboost
import shap
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from preprocess.feature_label_preparer import FeatureLabelPreparer
from run import get_labels_and_raw_feature_selections


class XGBoostFeatureExtractor:
    def __init__(self, model_file, feature_name, X):
        """

        :param model_file: the full path for the file containing xgboost model
        :param feature_name: pandas series containing name for features
        """
        self.model = joblib.load(model_file)
        self.feature_names = feature_name.values
        self.X = X

    def find_most_important_n_features(self, n, save_images=False, drug='tmp_drug'):
        xgboost.plot_importance(self.model)
        plt.title("xgboost.plot_importance(model)")
        plt.show()

        shap.initjs()
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X)

        # feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))[::-n]
        global_shap_values = np.abs(shap_values).mean(0)
        most_important_feature_indices = np.argsort(global_shap_values)[:-(n+1):-1]

        # shap.force_plot(explainer.expected_value, shap_values[0, :], self.X.iloc[0, :])
        # shap.force_plot(explainer.expected_value, shap_values[:50, :], self.X.iloc[:50, :])

        #shap.summary_plot(shap_values, self.X, max_display=n, plot_type="dot")
        # shap.summary_plot(shap_values, self.X, max_display=n, plot_type="compact_dot")
        shap.summary_plot(shap_values, self.X, max_display=n, plot_type="bar")
        #shap.summary_plot(shap_values, self.X, max_display=n, plot_type="violin")

        return self.X.columns[most_important_feature_indices], global_shap_values[most_important_feature_indices]


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

    xgb = XGBoostFeatureExtractor('/run/media/herkut/hdd-1/TB_genomes/truba/ar_detector_results_dataset-ii_20191205/best_models/xgboost_accuracy_phenotype_binary_snp_09_bcf_nu_indel_00_platypus_all/xgboost_Isoniazid.sav',
                                  raw_feature_matrix.columns,
                                  raw_feature_matrix)

    important_features, _ = xgb.find_most_important_n_features(10)
    print(important_features)
