import numpy as np

from models.logistic_regression import ARDetectorByLogisticRegression
from models.random_forest import ARDetectorByRandomForest
from models.svm import ARDetectorBySVMWithRBF, ARDetectorBySVMWithLinear

######################################################################
from preprocess.data_representation_preparer import DataRepresentationPreparer

target_drugs = ['Isoniazid', 'Rifampicin', 'Ethambutol', 'Pyrazinamide', 'Streptomycin', 'Ofloxacin', 'Amikacin', 'Ciprofloxacin', 'Moxifloxacin', 'Capreomycin', 'Kanamycin']
label_tags = 'phenotype'
TRADITIONAL_ML_SCORING = 'accuracy'
directory_containing_indexes = '/run/media/herkut/hdd-1/TB_genomes/features/dataset-1-train_test_indexes/'
TEST_SIZE = 0.2
######################################################################


class ModelManager:
    def __init__(self, models, data_representation='binary'):
        self.data_representation = data_representation

        # Set which models would be trained
        models_arr = models.split(',')
        self.enable_svm_linear = False
        self.enable_svm_rbf = False
        self.enable_rf = False
        self.enable_lr = False

        for model in models_arr:
            if model == 'svm_linear':
                self.enable_svm_linear = True
            if model == 'svm_rbf':
                self.enable_svm_rbf = True
            if model == 'rf':
                self.enable_rf = True
            if model == 'lr':
                self.enable_lr = True

    def train_and_test_models(self, results_directory, feature_selection, raw_feature_matrix, raw_labels):
        for i in range(len(target_drugs)):
            x, y = self.filter_out_nan(raw_feature_matrix, raw_labels[target_drugs[i]])

            tr_indexes = np.genfromtxt(directory_containing_indexes + target_drugs[i] + '_tr_indices.csv',
                                       delimiter=' ',
                                       dtype=np.int32)

            te_indexes = np.genfromtxt(directory_containing_indexes + target_drugs[i] + '_te_indices.csv',
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

            print("For the antibiotic " + target_drugs[i])
            print("Size of training dataset " + str(np.shape(x_train)))
            print("Size of test dataset " + str(np.shape(x_test)))

            #####################################
            #                                   #
            #           SVM with rbf            #
            #                                   #
            #####################################
            if self.enable_svm_rbf:
                ar_detector = ARDetectorBySVMWithRBF(results_directory,
                                                     feature_selection,
                                                     target_drugs[i],
                                                     label_tags=label_tags,
                                                     scoring=TRADITIONAL_ML_SCORING,
                                                     class_weights=class_weights)
                # train the model
                self.train_svm_with_rbf(ar_detector,
                                        x_train,
                                        y_train)
                # test the model
                ar_detector = ARDetectorBySVMWithRBF(results_directory,
                                                     feature_selection,
                                                     target_drugs[i],
                                                     label_tags=label_tags,
                                                     scoring=TRADITIONAL_ML_SCORING,
                                                     class_weights=class_weights)
                self.test_svm_with_rbf(ar_detector,
                                       x_test,
                                       y_test)
            #####################################
            #                                   #
            #         SVM with linear           #
            #                                   #
            #####################################
            if self.enable_svm_linear:
                ar_detector = ARDetectorBySVMWithLinear(results_directory,
                                                        feature_selection,
                                                        target_drugs[i],
                                                        label_tags=label_tags,
                                                        scoring=TRADITIONAL_ML_SCORING,
                                                        class_weights=class_weights)
                # train the model
                self.train_svm_with_rbf(ar_detector,
                                        x_train,
                                        y_train)
                # test the model
                ar_detector = ARDetectorBySVMWithLinear(results_directory,
                                                        feature_selection,
                                                        target_drugs[i],
                                                        label_tags=label_tags,
                                                        scoring=TRADITIONAL_ML_SCORING,
                                                        class_weights=class_weights)
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
                                                       scoring=TRADITIONAL_ML_SCORING,
                                                       class_weights=class_weights)
                # train the model
                self.train_random_forest(ar_detector,
                                         x_train,
                                         y_train)
                # test the model
                ar_detector = ARDetectorByRandomForest(results_directory,
                                                       feature_selection,
                                                       target_drugs[i],
                                                       label_tags=label_tags,
                                                       scoring=TRADITIONAL_ML_SCORING,
                                                       class_weights=class_weights)
                self.test_random_forest(ar_detector,
                                        x_test,
                                        y_test)

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
                                                             scoring=TRADITIONAL_ML_SCORING,
                                                             class_weights=class_weights)
                # train the model
                self.train_logistic_regression(ar_detector,
                                               x_train,
                                               y_train)
                # test the model
                ar_detector = ARDetectorByLogisticRegression(results_directory,
                                                             feature_selection,
                                                             target_drugs[i],
                                                             label_tags=label_tags,
                                                             scoring=TRADITIONAL_ML_SCORING,
                                                             class_weights=class_weights)
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

        ar_detector.tune_hyperparameters(param_grid, x_tr, y_tr)

        print(ar_detector._best_model)

    def train_svm_with_linear(self, ar_detector, x_tr, y_tr):
        # conduct svm model
        c_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

        param_grid = {'C': c_range}

        print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
        print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))
        #print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.tune_hyperparameters(param_grid, x_tr, y_tr)

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

        ar_detector.tune_hyperparameters(param_grid, x_tr, y_tr)

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

        ar_detector.tune_hyperparameters(param_grid, x_tr, y_tr)

        print(ar_detector._best_model)

    def test_svm_with_rbf(self, ar_detector, x_te, y_te):
        print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.load_model()
        ar_detector.test_model(x_te, y_te)

    def test_svm_with_linear(self, ar_detector, x_te, y_te):
        print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.load_model()
        ar_detector.test_model(x_te, y_te)

    def test_random_forest(self, ar_detector, x_te, y_te):
        print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.load_model()
        ar_detector.test_model(x_te, y_te)

    def test_logistic_regression(self, ar_detector, x_te, y_te):
        print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.load_model()
        ar_detector.test_model(x_te, y_te)
