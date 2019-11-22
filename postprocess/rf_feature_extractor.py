import os

import joblib
from tabulate import tabulate

from config import Config
from preprocess.feature_label_preparer import FeatureLabelPreparer
from run import get_labels_and_raw_feature_selections


class RandomForestFeatureExtractor:
    def __init__(self, model_file, feature_name):
        """

        :param model_file: the full path for the file containing scikit learn random forest model
        :param feature_name: pandas series containing name for features
        """
        self.model = joblib.load(model_file)
        self.feature_names = feature_name.values

    def find_most_important_n_features(self, n):
        headers = ['mutation', 'score']
        values = sorted(zip(self.feature_names, self.model.feature_importances_), key=lambda x: x[1] * -1)
        """
        f = open('/home/herkut/Desktop/rf_feature_Isoniazid.txt', 'w')
        f.write(tabulate(values[:n], headers=headers[:n], tablefmt='grid'))
        f.close()
        """
        return values[:n]


if __name__ == '__main__':
    raw = open('/home/herkut/Desktop/ar_detector/configurations/conf.yml')
    Config.initialize_configurations(raw)

    label_file, raw_feature_selections = get_labels_and_raw_feature_selections('dataset-ii')

    feature_selections = {}
    for k, v in raw_feature_selections.items():
        feature_selections['binary' + '_' + k] = v

    for k, v in feature_selections.items():
        print("Feature importance would be extacted for: " + k)
        raw_label_matrix = FeatureLabelPreparer.get_labels_from_file(os.path.join(Config.dataset_directory, label_file))
        raw_feature_matrix = FeatureLabelPreparer.get_feature_matrix_from_files(v)

    rf = RandomForestFeatureExtractor('/home/herkut/Desktop/truba/ar_detector_results_dataset-ii_20191118/best_models/rf_accuracy_phenotype_binary_snp_09_bcf_nu_indel_00_platypus_all/rf_Isoniazid.sav',
                                      raw_feature_matrix.columns)

    important_features = rf.find_most_important_n_features(100)



