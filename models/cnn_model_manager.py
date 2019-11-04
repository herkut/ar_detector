import json
import os

import numpy as np

from config import Config
from models.pytorch_models.ar_detector_cnn import ARDetectorCNN
from preprocess.feature_label_preparer import FeatureLabelPreparer
from utils.helper_functions import get_index_to_remove


class CNNModelManager:
    def __init__(self, models, dataset):
        self.dataset = dataset
        # Set which models would be trained
        self.models = models.split(',')
        self.cnn_models = []

        for model in self.models:
            if model.startswith('conv'):
                self.cnn_models.append(model)

        self.raw_data = {}

        for target_drug in Config.target_drugs:
            raw_label_matrix = FeatureLabelPreparer.get_labels_from_file(os.path.join(Config.dataset_directory,
                                                                                      'sorted_labels_dataset-ii.csv'))

            non_existing = []
            predefined_file_to_remove = ['8316-09', 'NL041']

            index_to_remove = get_index_to_remove(raw_label_matrix[target_drug])

            for ne in predefined_file_to_remove:
                if ne not in index_to_remove:
                    non_existing.append(ne)

            raw_label_matrix.drop(index_to_remove, inplace=True)
            raw_label_matrix.drop(non_existing, inplace=True)

            tr_indexes = np.genfromtxt(os.path.join(Config.dataset_index_directory + '_' + Config.target_dataset,
                                                    target_drug + '_tr_indices.csv'),
                                       delimiter=' ',
                                       dtype=str)

            te_indexes = np.genfromtxt(os.path.join(Config.dataset_index_directory + '_' + Config.target_dataset,
                                                    target_drug + '_te_indices.csv'),
                                       delimiter=' ',
                                       dtype=str)

            unique, counts = np.unique(raw_label_matrix[target_drug].values, return_counts=True)

            # class_weights = {0: counts[1] / (counts[0] + counts[1]), 1: counts[0] / (counts[0] + counts[1])}
            class_weights = {0: np.max(counts) / counts[0], 1: np.max(counts) / counts[1]}
            class_weights = np.array(list(class_weights.items()), dtype=np.float32)
            # xx = raw_label_matrix[tr_indexes].index
            tr_data = raw_label_matrix.loc[tr_indexes, target_drug]
            te_data = raw_label_matrix.loc[te_indexes, target_drug]
            self.raw_data[target_drug] = {'class_weights': class_weights,
                                          'tr_idx': tr_data.index,
                                          'tr_labels': tr_data.values,
                                          'te_idx': te_data.index,
                                          'te_labels': te_data.values}

    def tune_train_and_test_models(self):
        for i in range(len(Config.target_drugs)):
            for j in range(len(self.models)):
                model_name = self.models[j]
                ar_detector = ARDetectorCNN(Config.cnn_feature_size,
                                            Config.cnn_first_in_channel,
                                            Config.cnn_output_size,
                                            antibiotic_name=Config.target_drugs[i],
                                            model_name=model_name,
                                            class_weights=self.raw_data[Config.target_drugs[i]]['class_weights'])

                self.tune_hyperparameters_for_ar_detector(ar_detector,
                                                          Config.target_drugs[i])

                self.train_best_model(ar_detector,
                                      Config.target_drugs[i])

                self.test_ar_detector(ar_detector,
                                      self.raw_data[Config.target_drugs[i]]['te_idx'],
                                      self.raw_data[Config.target_drugs[i]]['te_labels'])

    def train_and_test_best_models(self):
        for i in range(len(Config.target_drugs)):
            for j in range(self.models):
                model_name = self.models[j]
                ar_detector = ARDetectorCNN(Config.cnn_feature_size,
                                            Config.cnn_first_in_channel,
                                            antibiotic_name=Config.target_drugs[i],
                                            model_name=model_name,
                                            class_weights=self.raw_data[Config.target_drugs[i]['class_weights']])
                self.train_best_model(ar_detector,
                                      Config.target_drugs[i])
                self.test_ar_detector(ar_detector,
                                      self.raw_data[Config.target_drugs[i]]['te_idx'],
                                      self.raw_data[Config.target_drugs[i]]['te_labels'])

    def test_best_models(self):
        for i in range(len(Config.target_drugs)):
            for j in range(self.models):
                model_name = self.models[j]
                ar_detector = ARDetectorCNN(Config.cnn_feature_size,
                                            Config.cnn_first_in_channel,
                                            antibiotic_name=Config.target_drugs[i],
                                            model_name=model_name,
                                            class_weights=self.raw_data[Config.target_drugs[i]['class_weights']])
                self.test_ar_detector(ar_detector,
                                      self.raw_data[Config.target_drugs[i]]['te_idx'],
                                      self.raw_data[Config.target_drugs[i]]['te_labels'])

    def tune_hyperparameters_for_ar_detector(self, ar_detector, target_drug):
        if not os.path.exists(os.path.join(Config.hyperparameter_grids_directory, ar_detector._model_name + '.json')):
            raise Exception('Hyperparameter grid could not be found for ' + ar_detector._model_name + ': ' + os.path.join(Config.hyperparameter_grids_directory, ar_detector._model_name + '.json'))

        with open(os.path.join(Config.hyperparameter_grids_directory, ar_detector._model_name + '.json')) as json_data:
            param_grid = json.load(json_data)

        ar_detector.tune_hyperparameters(param_grid,
                                         self.raw_data[target_drug]['tr_idx'],
                                         self.raw_data[target_drug]['tr_labels'])

    def train_best_model(self, ar_detector, target_drug):
        with open(os.path.join(ar_detector._results_directory,
                               'best_models',
                               ar_detector._target_directory,
                               ar_detector._model_name + '_' + ar_detector._antibiotic_name + '.json')) as fp:
            best_hyperparameters = json.load(fp)

        ar_detector.train_best_model(best_hyperparameters,
                                     self.raw_data[target_drug]['tr_idx'],
                                     self.raw_data[target_drug]['tr_labels'],
                                     self.raw_data[target_drug]['te_idx'],
                                     self.raw_data[target_drug]['te_labels'])

    def test_ar_detector(self, ar_detector, idx, labels):
        ar_detector.load_model()
        ar_detector.test_model(idx,
                               labels)
