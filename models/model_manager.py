import numpy as np
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split

from models.dnn import ArDetectorByDNN
from models.logistic_regression import ARDetectorByLogisticRegression
from models.random_forest import ARDetectorByRandomForest
from models.svm_rbf import ARDetectorBySVMWithRBF


######################################################################
from preprocess.data_representation_preparer import DataRepresentationPreparer

target_drugs = ['Isoniazid', 'Rifampicin', 'Ethambutol', 'Pyrazinamide', 'Streptomycin', 'Ofloxacin', 'Amikacin', 'Ciprofloxacin', 'Moxifloxacin', 'Capreomycin', 'Kanamycin']
label_tags = 'phenotype'
TRADITIONAL_ML_SCORING = 'f1'
TEST_SIZE = 0.2
######################################################################


class ModelManager:
    def __init__(self, models, data_representation='binary'):
        self.data_representation = data_representation

        # Set which models would be trained
        models_arr = models.split(',')
        self.enable_svm = False
        self.enable_rf = False
        self.enable_dnn = False
        self.enable_lr = False

        for model in models_arr:
            if model == 'svm':
                self.enable_svm = True
            if model == 'rf':
                self.enable_rf = True
            if model == 'dnn':
                self.enable_dnn = True
            if model == 'lr':
                self.enable_lr = True

    def train_and_test_models(self, results_directory, feature_selection, raw_feature_matrix, raw_labels):
        for i in range(len(target_drugs)):
            x, y = self.filter_out_nan(raw_feature_matrix, raw_labels[target_drugs[i]])

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
            x = x.values
            y = y.values
            x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=TEST_SIZE, random_state=0)

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
                self.test_random_forest(ar_detector,
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
                                              label_tags=label_tags
                                              )

                self.train_dnn(ar_detector, x_train, y_train)

                self.test_dnn(ar_detector, x_test, y_test)

            #####################################
            #                                   #
            #        Logistic Regression        #
            #                                   #
            #####################################
            if self.enable_lr:
                ar_detector = ARDetectorByLogisticRegression(results_directory,
                                                             feature_selection,
                                                             target_drugs[i],
                                                             label_tags=label_tags,
                                                             scoring=TRADITIONAL_ML_SCORING)
                # train the model
                self.train_logistic_regression(ar_detector,
                                               x_train,
                                               y_train)
                # test the model
                ar_detector = ARDetectorByLogisticRegression(results_directory,
                                                             feature_selection,
                                                             target_drugs[i],
                                                             label_tags=label_tags,
                                                             scoring=TRADITIONAL_ML_SCORING)
                self.test_logistic_regression(ar_detector,
                                              x_test,
                                              y_test)

    def filter_out_nan(self, x, y):
        index_to_remove = y[y.isna() == True].index
        #index_to_remove = np.argwhere(np.isnan(y)).values

        xx = x.drop(index_to_remove, inplace=False)
        yy = y.drop(index_to_remove, inplace=False)

        #xx = np.delete(x, index_to_remove, axis=0)
        #yy = np.delete(y, index_to_remove, axis=0)

        return xx, yy

    def train_svm_with_rbf(self, ar_detector, x_tr, y_tr):
        # conduct svm model
        c_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        gamma_range = [0.001, 0.1, 1, 10, 100]

        param_grid = {'C': c_range, 'gamma': gamma_range}

        print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
        print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))
        #print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.initialize_train_dataset(x_tr, y_tr)

        ar_detector.tune_hyperparameters(param_grid)

        print(ar_detector._best_model)

    def train_random_forest(self, ar_detector, x_tr, y_tr):
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

        print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
        print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))
        #print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.initialize_train_dataset(x_tr, y_tr)

        ar_detector.tune_hyperparameters(param_grid)

        print(ar_detector._best_model)

    def train_logistic_regression(self, ar_detector, x_tr, y_tr):
        c_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        # 'none', 'elasticnet', 'l1',
        penalty = ['l2']
        solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

        param_grid = {'C': c_range,
                      'penalty': penalty,
                      'solver': solver}

        print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
        print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))
        #print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.initialize_train_dataset(x_tr, y_tr)

        ar_detector.tune_hyperparameters(param_grid)

        print(ar_detector._best_model)

    def train_dnn(self, ar_detector, x_tr, y_tr):
        # Optimizers to be tried are selected according to Karpathy's following blog page: https://medium.com/@karpathy/a-peek-at-trends-in-machine-learning-ab8a1085a106
        # hidden units and activation functions elements must be the same sized because they would create a hidden layer together
        param_grid = dict(hidden_units=[[128], [1024], [4096]],
                          activation_functions=[['relu'], ['tanh'], ['linear']],
                          epochs=[100],
                          batch_size=[50, 100],
                          optimizer=['RMSprop', 'Adam', 'Adagrad', 'Adadelta'],
                          dropout_rate=[0.0, 0.25, 0.5],
                          batch_normalization_required=[True])

        """
        param_grid = dict(hidden_units=[[256]],
                          activation_functions=[['relu'], ['linear']],
                          epochs=[100],
                          batch_size=[100],
                          optimizer=['Adam', 'RMSprop'],
                          dropout_rate=[0.25],
                          batch_normalization_required=[True])
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
        ar_detector.load_model()
        ar_detector.test_model()

    def test_logistic_regression(self, ar_detector, x_te, y_te):
        print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.initialize_test_dataset(x_te, y_te)
        ar_detector.load_model()
        ar_detector.test_model()
