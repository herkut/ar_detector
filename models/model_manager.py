import numpy as np
from sklearn.model_selection import train_test_split

from models.dnn import ArDetectorByDNN
from models.random_forest import ARDetectorByRandomForest
from models.svm_rbf import ARDetectorBySVMWithRBF


######################################################################
#target_drugs = ['Isoniazid', 'Rifampicin', 'Ethambutol', 'Pyrazinamide', 'Streptomycin', 'Ofloxacin', 'Amikacin']
target_drugs = ['Streptomycin', 'Ofloxacin', 'Amikacin']
label_tags = 'phenotype'
TRADITIONAL_ML_SCORING = 'accuracy'
TEST_SIZE = 0.3
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
            x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=TEST_SIZE)
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
                                        x_train,
                                        y_train)
                # test the model
                ar_detector = ARDetectorBySVMWithRBF(results_directory,
                                                     feature_selection,
                                                     target_drugs[i],
                                                     label_tags=label_tags,
                                                     scoring=TRADITIONAL_ML_SCORING)
                self.test_svm_with_rbf(ar_detector,
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
                                         x_train,
                                         y_train)
                # test the model
                ar_detector = ARDetectorByRandomForest(results_directory,
                                                       feature_selection,
                                                       target_drugs[i],
                                                       label_tags=label_tags,
                                                       scoring=TRADITIONAL_ML_SCORING)
                self.test_svm_with_rbf(ar_detector,
                                       x_test,
                                       y_test)

            #####################################
            #                                   #
            #        Deep Neural Network        #
            #                                   #
            #####################################
            if self.enable_dnn:
                ar_detector = ArDetectorByDNN(results_directory,
                                              feature_selection,
                                              target_drugs[i],
                                              np.shape(x_train)[1],
                                              [256, 128, 64],
                                              ['relu', 'relu', 'relu'],
                                              label_tags=label_tags
                                              )

                self.train_dnn(ar_detector, x_train, y_train)

                self.test_dnn(ar_detector, x_test, y_test)

    def filter_out_nan(self, x, y):
        index_to_remove = np.argwhere(np.isnan(y))

        xx = np.delete(x, index_to_remove, axis=0)
        yy = np.delete(y, index_to_remove, axis=0)

        return xx, yy

    def train_svm_with_rbf(self, ar_detector, x_tr, y_tr):
        # conduct svm model
        c_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        # c_range = [1]
        gamma_range = [0.001, 0.1, 1, 10, 100]
        # gamma_range = [1]

        print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
        print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))
        #print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.initialize_train_dataset(x_tr, y_tr)

        ar_detector.tune_hyperparameters(c_range, gamma_range)

        print(ar_detector._best_model)

    def train_random_forest(self, ar_detector, x_tr, y_tr):
        # bootstrap = [True, False]
        n_estimators = [int(x) for x in np.linspace(start=100, stop=500, num=100)]
        max_features = ['sqrt', 'log2', None]

        print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
        print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))
        #print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.initialize_train_dataset(x_tr, y_tr)

        ar_detector.tune_hyperparameters(n_estimators, max_features)

        print(ar_detector._best_model)

    def train_dnn(self, ar_detector, x_tr, y_tr):
        # Optimizers to be tried are selected according to Karpathy's following blog page: https://medium.com/@karpathy/a-peek-at-trends-in-machine-learning-ab8a1085a106
        param_grid = dict(epochs=[50],
                          batch_size=[1, 50, 100],
                          optimizer=['RMSprop', 'Adam', 'Adagrad', 'Adadelta'],
                          dropout_rate=[0.0, 0.5, 0.9],
                          batch_normalization_required=[True])
        """
        param_grid = dict(epochs=[50, 100],
                          batch_size=[100],
                          optimizer=['Adam', 'RMSprop'],
                          dropout_rate=[0.9],
                          batch_normalization_required=[False, True])
        """
        print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
        print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))

        ar_detector.initialize_train_dataset(x_tr, y_tr)
        ar_detector.tune_hyperparameters(param_grid)

    def test_svm_with_rbf(self, ar_detector, x_te, y_te):
        print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.initialize_test_dataset(x_te, y_te)
        ar_detector.load_model()
        ar_detector.test_model()

    def test_random_forest(self, ar_detector, x_te, y_te):
        print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.initialize_test_dataset(x_te, y_te)
        ar_detector.load_model()
        ar_detector.test_model()

    def test_dnn(self, ar_detector, x_te, y_te):
        print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.initialize_test_dataset(x_te, y_te)
        # ar_detector.load_model()
        ar_detector.test_model()
