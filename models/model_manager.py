import json
import os

import numpy as np

from config import Config
from models.logistic_regression import ARDetectorByLogisticRegression
from models.pytorch_models.ar_detector_dnn import ARDetectorDNN
from models.random_forest import ARDetectorByRandomForest
from models.svm import ARDetectorBySVMWithRBF, ARDetectorBySVMWithLinear
from preprocess.data_representation_preparer import DataRepresentationPreparer


class ModelManager:
    def __init__(self, models, dataset, data_representation='binary'):
        self.data_representation = data_representation
        self.dataset = dataset
        # Set which models would be trained
        self.models = models.split(',')
        self.dnn_models = []

        for model in self.models:
            if model.startswith('dnn'):
                self.dnn_models.append(model)

    def tune_train_and_test_models(self, feature_selection, raw_feature_matrix, raw_labels):
        for i in range(len(Config.target_drugs)):
            x, y = self.filter_out_nan(raw_feature_matrix, raw_labels[Config.target_drugs[i]])

            tr_indexes = np.genfromtxt(os.path.join(Config.dataset_index_directory, Config.target_drugs[i] + '_tr_indices.csv'),
                                       delimiter=' ',
                                       dtype=np.int32)

            te_indexes = np.genfromtxt(os.path.join(Config.dataset_index_directory, Config.target_drugs[i] + '_te_indices.csv'),
                                       delimiter=' ',
                                       dtype=np.int32)

            # Random state is used to make train and test split the same on each iteration
            if self.data_representation == 'tfidf':
                x = DataRepresentationPreparer.update_feature_matrix_with_tf_idf(x)
            elif self.data_representation == 'tfrf':
                x = DataRepresentationPreparer.update_feature_matrix_with_tf_rf(x, y)
            elif self.data_representation == 'bm25tfidf':
                x = DataRepresentationPreparer.update_feature_matrix_with_bm25_tf_idf(x)
            elif self.data_representation == 'bm25tfrf':
                x = DataRepresentationPreparer.update_feature_matrix_with_bm25_tf_rf(x, y)
            else:
                # Assumed binary data representation would be used
                pass

            # Update weights of features if necessary
            x_train = x.loc[tr_indexes].values
            y_train = y.loc[tr_indexes].values
            x_test = x.loc[te_indexes].values
            y_test = y.loc[te_indexes].values

            x = x.values
            y = y.values

            class_weights_arr = {}

            unique, counts = np.unique(y, return_counts=True)

            class_weights = {0: counts[1] / (counts[0] + counts[1]), 1: counts[0] / (counts[0] + counts[1])}

            print("For the antibiotic " + Config.target_drugs[i])
            print("Size of training dataset " + str(np.shape(x_train)))
            print("Size of test dataset " + str(np.shape(x_test)))

            for model in self.models:
                if model == 'svm_rbf':
                    ar_detector = ARDetectorBySVMWithRBF(feature_selection,
                                                         self.dataset,
                                                         Config.target_drugs[i],
                                                         class_weights=class_weights)

                elif model == 'svm_linear':
                    ar_detector = ARDetectorBySVMWithLinear(feature_selection,
                                                            self.dataset,
                                                            Config.target_drugs[i],
                                                            class_weights=class_weights)

                elif model == 'rf':
                    ar_detector = ARDetectorByRandomForest(feature_selection,
                                                           self.dataset,
                                                           Config.target_drugs[i],
                                                           class_weights=class_weights)

                elif model == 'lr':
                    ar_detector = ARDetectorByLogisticRegression(feature_selection,
                                                                 self.dataset,
                                                                 Config.target_drugs[i],
                                                                 class_weights=class_weights)

                elif model in self.dnn_models:
                    # convert class weight into numpy matrix
                    class_weights_numpy = np.array(list(class_weights.items()), dtype=np.float32)
                    ar_detector = ARDetectorDNN(feature_selection,
                                                self.dataset,
                                                antibiotic_name=Config.target_drugs[i],
                                                model_name=model,
                                                feature_size=x_train.shape[1],
                                                class_weights=class_weights_numpy)

                else:
                    raise Exception('Unknown model: ' + model)
                # tune hyperparameters for the model
                self.tune_hyperparameters_for_ar_detector(ar_detector,
                                                          x_train,
                                                          y_train)

                # train model with best hyperparameters
                self.train_best_model(ar_detector,
                                      x_train,
                                      y_train,
                                      x_test,
                                      y_test)

                # test pretrained model with the best hyperparameters
                self.test_ar_detector(ar_detector,
                                      x_test,
                                      y_test)

    def train_and_test_best_models(self, feature_selection, raw_feature_matrix, raw_labels):
        for i in range(len(Config.target_drugs)):
            x, y = self.filter_out_nan(raw_feature_matrix, raw_labels[Config.target_drugs[i]])

            tr_indexes = np.genfromtxt(os.path.join(Config.dataset_index_directory, Config.target_drugs[i] + '_tr_indices.csv'),
                                       delimiter=' ',
                                       dtype=np.int32)

            te_indexes = np.genfromtxt(os.path.join(Config.dataset_index_directory, Config.target_drugs[i] + '_te_indices.csv'),
                                       delimiter=' ',
                                       dtype=np.int32)

            # Random state is used to make train and test split the same on each iteration
            if self.data_representation == 'tfidf':
                x = DataRepresentationPreparer.update_feature_matrix_with_tf_idf(x)
            elif self.data_representation == 'tfrf':
                x = DataRepresentationPreparer.update_feature_matrix_with_tf_rf(x, y)
            elif self.data_representation == 'bm25tfidf':
                x = DataRepresentationPreparer.update_feature_matrix_with_bm25_tf_idf(x)
            elif self.data_representation == 'bm25tfrf':
                x = DataRepresentationPreparer.update_feature_matrix_with_bm25_tf_rf(x, y)
            else:
                # Assumed binary data representation would be used
                pass

            # Update weights of features if necessary
            x_train = x.loc[tr_indexes].values
            y_train = y.loc[tr_indexes].values
            x_test = x.loc[te_indexes].values
            y_test = y.loc[te_indexes].values

            x = x.values
            y = y.values

            class_weights_arr = {}

            unique, counts = np.unique(y, return_counts=True)

            class_weights = {0: counts[1] / (counts[0] + counts[1]), 1: counts[0] / (counts[0] + counts[1])}

            print("For the antibiotic " + Config.target_drugs[i])
            print("Size of training dataset " + str(np.shape(x_train)))
            print("Size of test dataset " + str(np.shape(x_test)))

            for model in self.models:
                if model == 'svm_rbf':
                    ar_detector = ARDetectorBySVMWithRBF(feature_selection,
                                                         Config.target_drugs[i],
                                                         class_weights=class_weights)

                elif model == 'svm_linear':
                    ar_detector = ARDetectorBySVMWithLinear(feature_selection,
                                                            Config.target_drugs[i],
                                                            class_weights=class_weights)

                elif model == 'rf':
                    ar_detector = ARDetectorByRandomForest(feature_selection,
                                                           Config.target_drugs[i],
                                                           class_weights=class_weights)

                elif model == 'lr':
                    ar_detector = ARDetectorByLogisticRegression(feature_selection,
                                                                 Config.target_drugs[i],
                                                                 class_weights=class_weights)

                elif model in self.dnn_models:
                    # convert class weight into numpy matrix
                    class_weights_numpy = np.array(list(class_weights.items()), dtype=np.float32)
                    ar_detector = ARDetectorDNN(feature_selection,
                                                antibiotic_name=Config.target_drugs[i],
                                                model_name=model,
                                                feature_size=x_train.shape[1],
                                                class_weights=class_weights_numpy)

                else:
                    raise Exception('Unknown model: ' + model)

                # train model with best hyperparameters
                self.train_best_model(ar_detector,
                                      x_train,
                                      y_train,
                                      x_test,
                                      y_test)

                # test pretrained model with the best hyperparameters
                self.test_ar_detector(ar_detector,
                                      x_test,
                                      y_test)
    def test_best_models(self, feature_selection, raw_feature_matrix, raw_labels):
        for i in range(len(Config.target_drugs)):
            x, y = self.filter_out_nan(raw_feature_matrix, raw_labels[Config.target_drugs[i]])

            tr_indexes = np.genfromtxt(os.path.join(Config.dataset_index_directory, Config.target_drugs[i] + '_tr_indices.csv'),
                                       delimiter=' ',
                                       dtype=np.int32)

            te_indexes = np.genfromtxt(os.path.join(Config.dataset_index_directory, Config.target_drugs[i] + '_te_indices.csv'),
                                       delimiter=' ',
                                       dtype=np.int32)

            # Random state is used to make train and test split the same on each iteration
            if self.data_representation == 'tfidf':
                x = DataRepresentationPreparer.update_feature_matrix_with_tf_idf(x)
            elif self.data_representation == 'tfrf':
                x = DataRepresentationPreparer.update_feature_matrix_with_tf_rf(x, y)
            elif self.data_representation == 'bm25tfidf':
                x = DataRepresentationPreparer.update_feature_matrix_with_bm25_tf_idf(x)
            elif self.data_representation == 'bm25tfrf':
                x = DataRepresentationPreparer.update_feature_matrix_with_bm25_tf_rf(x, y)
            else:
                # Assumed binary data representation would be used
                pass

            # Update weights of features if necessary
            x_train = x.loc[tr_indexes].values
            y_train = y.loc[tr_indexes].values
            x_test = x.loc[te_indexes].values
            y_test = y.loc[te_indexes].values

            x = x.values
            y = y.values

            class_weights_arr = {}

            unique, counts = np.unique(y, return_counts=True)

            class_weights = {0: counts[1] / (counts[0] + counts[1]), 1: counts[0] / (counts[0] + counts[1])}

            print("For the antibiotic " + Config.target_drugs[i])
            print("Size of training dataset " + str(np.shape(x_train)))
            print("Size of test dataset " + str(np.shape(x_test)))

            for model in self.models:
                if model == 'svm_rbf':
                    ar_detector = ARDetectorBySVMWithRBF(feature_selection,
                                                         Config.target_drugs[i],
                                                         class_weights=class_weights)

                elif model == 'svm_linear':
                    ar_detector = ARDetectorBySVMWithLinear(feature_selection,
                                                            Config.target_drugs[i],
                                                            class_weights=class_weights)

                elif model == 'rf':
                    ar_detector = ARDetectorByRandomForest(feature_selection,
                                                           Config.target_drugs[i],
                                                           class_weights=class_weights)

                elif model == 'lr':
                    ar_detector = ARDetectorByLogisticRegression(feature_selection,
                                                                 Config.target_drugs[i],
                                                                 class_weights=class_weights)

                elif model in self.dnn_models:
                    # convert class weight into numpy matrix
                    class_weights_numpy = np.array(list(class_weights.items()), dtype=np.float32)
                    ar_detector = ARDetectorDNN(feature_selection,
                                                antibiotic_name=Config.target_drugs[i],
                                                model_name=model,
                                                feature_size=x_train.shape[1],
                                                class_weights=class_weights_numpy)

                else:
                    raise Exception('Unknown model: ' + model)

                # test pretrained model stored in the best models directory
                self.test_ar_detector(ar_detector,
                                      x_test,
                                      y_test)

    def filter_out_nan(self, x, y):
        index_to_remove = y[y.isna() == True].index

        xx = x.drop(index_to_remove, inplace=False)
        yy = y.drop(index_to_remove, inplace=False)

        return xx, yy

    def tune_hyperparameters_for_ar_detector(self, ar_detector, x_tr, y_tr):
        if not os.path.exists(os.path.join(Config.hyperparameter_grids_directory, ar_detector._model_name + '.json')):
            raise Exception('Hyperparameter grid could not be found for ' + ar_detector._model_name + ': ' + os.path.join(Config.hyperparameter_grids_directory, ar_detector._model_name + '.json'))

        with open(os.path.join(Config.hyperparameter_grids_directory, ar_detector._model_name + '.json')) as json_data:
            param_grid = json.load(json_data)

        print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
        print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))

        ar_detector.tune_hyperparameters(param_grid, x_tr, y_tr)

        print(ar_detector._best_model)

    def train_best_model(self, ar_detector, x_tr, y_tr, x_te, y_te):
        with open(os.path.join(Config.results_directory,
                               'best_models',
                               ar_detector._target_directory,
                               ar_detector._model_name + '_' + ar_detector._antibiotic_name + '.json')) as fp:
            best_hyperparameters = json.load(fp)

        ar_detector.train_best_model(best_hyperparameters, x_tr, y_tr, x_te, y_te)

    def test_ar_detector(self, ar_detector, x_te, y_te):
        print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.load_model()
        ar_detector.test_model(x_te, y_te)
