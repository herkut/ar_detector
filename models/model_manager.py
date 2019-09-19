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
    def __init__(self, models, data_representation='binary'):
        self.data_representation = data_representation

        # Set which models would be trained
        models_arr = models.split(',')
        self.enable_svm_linear = False
        self.enable_svm_rbf = False
        self.enable_rf = False
        self.enable_lr = False
        self.dnn_models = []

        for model in models_arr:
            if model == 'svm_linear':
                self.enable_svm_linear = True
            if model == 'svm_rbf':
                self.enable_svm_rbf = True
            if model == 'rf':
                self.enable_rf = True
            if model == 'lr':
                self.enable_lr = True
            if model.startswith('dnn'):
                self.dnn_models.append(model)

    def train_and_test_models(self, feature_selection, raw_feature_matrix, raw_labels):
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

            #####################################
            #                                   #
            #           SVM with rbf            #
            #                                   #
            #####################################
            if self.enable_svm_rbf:
                ar_detector = ARDetectorBySVMWithRBF(feature_selection,
                                                     Config.target_drugs[i],
                                                     class_weights=class_weights)
                # train the model
                self.train_svm_with_rbf(ar_detector,
                                        x_train,
                                        y_train)
                # test the model
                ar_detector = ARDetectorBySVMWithRBF(feature_selection,
                                                     Config.target_drugs[i],
                                                     class_weights=class_weights)
                self.test_ar_detector(ar_detector,
                                      x_test,
                                      y_test)
            #####################################
            #                                   #
            #         SVM with linear           #
            #                                   #
            #####################################
            if self.enable_svm_linear:
                ar_detector = ARDetectorBySVMWithLinear(feature_selection,
                                                        Config.target_drugs[i],
                                                        class_weights=class_weights)
                # train the model
                self.train_svm_with_rbf(ar_detector,
                                        x_train,
                                        y_train)
                # test the model
                ar_detector = ARDetectorBySVMWithLinear(feature_selection,
                                                        Config.target_drugs[i],
                                                        class_weights=class_weights)
                self.test_ar_detector(ar_detector,
                                      x_test,
                                      y_test)

            #####################################
            #                                   #
            #           Random Forest           #
            #                                   #
            #####################################
            if self.enable_rf:
                ar_detector = ARDetectorByRandomForest(feature_selection,
                                                       Config.target_drugs[i],
                                                       class_weights=class_weights)
                # train the model
                self.train_random_forest(ar_detector,
                                         x_train,
                                         y_train)
                # test the model
                ar_detector = ARDetectorByRandomForest(feature_selection,
                                                       Config.target_drugs[i],
                                                       class_weights=class_weights)
                self.test_ar_detector(ar_detector,
                                      x_test,
                                      y_test)

            #####################################
            #                                   #
            #        Logistic Regression        #
            #                                   #
            #####################################
            if self.enable_lr:
                ar_detector = ARDetectorByLogisticRegression(feature_selection,
                                                             Config.target_drugs[i],
                                                             class_weights=class_weights)
                # train the model
                self.train_logistic_regression(ar_detector,
                                               x_train,
                                               y_train)
                # test the model
                ar_detector = ARDetectorByLogisticRegression(feature_selection,
                                                             Config.target_drugs[i],
                                                             class_weights=class_weights)
                self.test_ar_detector(ar_detector,
                                      x_test,
                                      y_test)

            #####################################
            #                                   #
            #               DNN                 #
            #                                   #
            #####################################
            if self.dnn_models:
                # convert class weight into numpy matrix
                class_weights_numpy = np.array(list(class_weights.items()), dtype=np.float32)
                for dnn_model in self.dnn_models:
                    ar_detector = ARDetectorDNN(feature_selection,
                                                Config.target_drugs[i],
                                                model_name=dnn_model,
                                                class_weights=class_weights_numpy)
                    self.train_ar_detector(ar_detector, x_train, y_train)

    def filter_out_nan(self, x, y):
        index_to_remove = y[y.isna() == True].index

        xx = x.drop(index_to_remove, inplace=False)
        yy = y.drop(index_to_remove, inplace=False)

        return xx, yy

    def train_ar_detector(self, ar_detector, x_tr, y_tr):
        if not os.path.exists(os.path.join(Config.hyperparameter_grids_directory, ar_detector._model_name + '.json')):
            raise Exception('Hyperparameter grid could not be found for svm rbf: ' + os.path.join(Config.hyperparameter_grids_directory, ar_detector._model_name + '.json'))

        with open(os.path.join(Config.hyperparameter_grids_directory, ar_detector._model_name + '.json')) as json_data:
            param_grid = json.load(json_data)

        print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
        print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))

        ar_detector.tune_hyperparameters(param_grid, x_tr, y_tr)

        print(ar_detector._best_model)


    def train_svm_with_rbf(self, ar_detector, x_tr, y_tr):
        # conduct svm model
        if not os.path.exists(os.path.join(Config.hyperparameter_grids_directory, 'svm_rbf.json')):
            raise Exception('Hyperparameter grid could not be found for svm rbf: ' + os.path.join(Config.hyperparameter_grids_directory, 'svm_rbf.json'))

        with open(os.path.join(Config.hyperparameter_grids_directory, 'svm_rbf.json')) as json_data:
            param_grid = json.load(json_data)

        print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
        print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))

        ar_detector.tune_hyperparameters(param_grid, x_tr, y_tr)

        print(ar_detector._best_model)

    def train_svm_with_linear(self, ar_detector, x_tr, y_tr):
        # conduct svm model
        if not os.path.exists(os.path.join(Config.hyperparameter_grids_directory, 'svm_linear.json')):
            raise Exception('Hyperparameter grid could not be found for svm linear: ' + os.path.join(Config.hyperparameter_grids_directory, 'svm_linear.json'))

        with open(os.path.join(Config.hyperparameter_grids_directory, 'svm_linear.json')) as json_data:
            param_grid = json.load(json_data)

        print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
        print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))

        ar_detector.tune_hyperparameters(param_grid, x_tr, y_tr)

        print(ar_detector._best_model)

    def train_random_forest(self, ar_detector, x_tr, y_tr):
        """
        # bootstrap = [True, False]
        n_estimators = [100, 250, 500, 1000]
        max_features = ['sqrt', 'log2', None]
        bootstrap = None
        max_depth = None

        param_grid = {'n_estimators': n_estimators, 'max_features': max_features}

        if bootstrap is not None:
            param_grid['bootstrap'] = bootstrap

        if max_depth is not None:
            param_grid['max_depth'] = max_depth

        """
        if not os.path.exists(os.path.join(Config.hyperparameter_grids_directory, 'rf.json')):
            raise Exception('Hyperparameter grid could not be found for rf: ' + os.path.join(Config.hyperparameter_grids_directory, 'rf.json'))

        with open(os.path.join(Config.hyperparameter_grids_directory, 'rf.json')) as json_data:
            param_grid = json.load(json_data)

        print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
        print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))

        ar_detector.tune_hyperparameters(param_grid, x_tr, y_tr)

        print(ar_detector._best_model)

    def train_logistic_regression(self, ar_detector, x_tr, y_tr):
        if not os.path.exists(os.path.join(Config.hyperparameter_grids_directory, 'lr.json')):
            raise Exception('Hyperparameter grid could not be found for lr: ' + os.path.join(Config.hyperparameter_grids_directory, 'lr.json'))

        with open(os.path.join(Config.hyperparameter_grids_directory, 'lr.json')) as json_data:
            param_grid = json.load(json_data)

        print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
        print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))

        ar_detector.tune_hyperparameters(param_grid, x_tr, y_tr)

        print(ar_detector._best_model)

    def test_ar_detector(self, ar_detector, x_te, y_te):
        print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.load_model()
        ar_detector.test_model(x_te, y_te)
