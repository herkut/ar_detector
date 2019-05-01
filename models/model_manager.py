import numpy as np
from sklearn.model_selection import train_test_split

from models.random_forest import ARDetectorByRandomForest
from models.svm_rbf import ARDetectorBySVMWithRBF


######################################################################
target_drugs = ['Isoniazid', 'Rifampicin', 'Ethambutol', 'Pyrazinamide', 'Streptomycin', 'Ofloxacin', 'Amikacin']
label_tags = 'phenotype'
TRADITIONAL_ML_SCORING = 'f1'
######################################################################


class ModelManager:
    def __init__(self, models):
        # Set which models would be trained
        models_arr = models.split(',')
        for model in models_arr:
            if model == 'svm':
                self.enable_svm = True
            else:
                self.enable_svm = False
            if model == 'rf':
                self.enable_rf = True
            else:
                self.enable_rf = False
            if model == 'dnn':
                self.enable_dnn = True
            else:
                self.enable_dnn = False

    def train_and_test_models(self, results_directory, feature_selection, raw_feature_matrix, raw_labels):
        for i in range(len(target_drugs)):
            x, y = self.filter_out_nan(raw_feature_matrix, raw_labels[:, i])
            x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=raw_labels, test_size=0.3)
            print("For the antibiotic " + target_drugs[i])
            print("Size of training dataset " + str(np.shape(x_train)))
            print("Size of test dataset " + str(np.shape(x_test)))
            #####################################
            #                                   #
            #           SVM with rbf            #
            #                                   #
            #####################################
            if self.enable_svm:
                ar_detector = ARDetectorBySVMWithRBF(results_directory,
                                                     feature_selection,
                                                     target_drugs[i],
                                                     label_tags=label_tags,
                                                     scoring=TRADITIONAL_ML_SCORING)
                # train the model
                self.train_svm_with_rbf(ar_detector,
                                        i,
                                        x_train,
                                        y_train)
                # test the model
                ar_detector = ARDetectorBySVMWithRBF(results_directory,
                                                     feature_selection,
                                                     target_drugs[i],
                                                     label_tags=label_tags,
                                                     scoring=TRADITIONAL_ML_SCORING)
                self.test_svm_with_rbf(ar_detector,
                                       i,
                                       x_test,
                                       y_test)

            #####################################
            #                                   #
            #           Random Forest           #
            #                                   #
            #####################################
            if self.enable_rf:
                ar_detector = ARDetectorByRandomForest(results_directory,
                                                       feature_selection,
                                                       target_drugs[i],
                                                       label_tags=label_tags,
                                                       scoring=TRADITIONAL_ML_SCORING)
                # train the model
                self.train_random_forest(ar_detector,
                                         i,
                                         x_train,
                                         y_train)
                # test the model
                ar_detector = ARDetectorByRandomForest(results_directory,
                                                       feature_selection,
                                                       target_drugs[i],
                                                       label_tags=label_tags,
                                                       scoring=TRADITIONAL_ML_SCORING)
                self.test_svm_with_rbf(ar_detector,
                                       i,
                                       x_test,
                                       y_test)

            #####################################
            #                                   #
            #        Deep Neural Network        #
            #                                   #
            #####################################
            if self.enable_dnn:
                pass

    def filter_out_nan(self, x, y):
        index_to_remove = np.argwhere(np.isnan(y))

        xx = np.delete(x, index_to_remove, axis=0)
        yy = np.delete(y, index_to_remove, axis=0)

        return xx, yy

    def train_svm_with_rbf(self, ar_detector, index_of_antibiotic, feature_matrix_training, labels_matrix_training):
        # conduct svm model
        c_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        # c_range = [1]
        gamma_range = [0.001, 0.1, 1, 10, 100]
        # gamma_range = [1]
        x_tr, y_tr = self.filter_out_nan(feature_matrix_training, labels_matrix_training[:, index_of_antibiotic])
        #x_te, y_te = self.filter_out_nan(feature_matrix_test, labels_matrix_test[:, index_of_antibiotic])

        print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
        print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))
        #print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.initialize_train_dataset(x_tr, y_tr)

        ar_detector.tune_hyperparameters(c_range, gamma_range)

        print(ar_detector._best_model)

    def train_random_forest(self, ar_detector, index_of_antibiotic, feature_matrix_training, labels_matrix_training):
        # bootstrap = [True, False]
        n_estimators = [int(x) for x in np.linspace(start=100, stop=500, num=100)]
        max_features = ['sqrt', 'log2', None]

        x_tr, y_tr = self.filter_out_nan(feature_matrix_training, labels_matrix_training[:, index_of_antibiotic])
        #x_te, y_te = self.filter_out_nan(feature_matrix_test, labels_matrix_test[:, index_of_antibiotic])

        print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
        print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))
        #print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.initialize_train_datasets(x_tr, y_tr)

        ar_detector.tune_hyperparameters(n_estimators, max_features)

        print(ar_detector._best_model)

    def train_dnn(self, ar_detector, index_of_antibiotic, feature_matrix_training, labels_matrix_training, feature_matrix_test, labels_matrix_test):
        pass

    def test_svm_with_rbf(self, ar_detector, index_of_antibiotic, feature_matrix_test, labels_matrix_test):
        #x_tr, y_tr = self.filter_out_nan(feature_matrix_training, labels_matrix_training[:, index_of_antibiotic])
        x_te, y_te = self.filter_out_nan(feature_matrix_test, labels_matrix_test[:, index_of_antibiotic])

        print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.initialize_test_datasets(x_te, y_te)
        ar_detector.load_model()
        ar_detector.test_model()

    def test_random_forest(self, ar_detector, index_of_antibiotic, feature_matrix_test, labels_matrix_test):
        #x_tr, y_tr = self.filter_out_nan(feature_matrix_training, labels_matrix_training[:, index_of_antibiotic])
        x_te, y_te = self.filter_out_nan(feature_matrix_test, labels_matrix_test[:, index_of_antibiotic])

        print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.initialize_test_datasets(x_te, y_te)
        ar_detector.load_model()
        ar_detector.test_model()

    def test_dnn(self, ar_detector, index_of_antibiotic, feature_matrix_training, labels_matrix_training, feature_matrix_test, labels_matrix_test):
        pass