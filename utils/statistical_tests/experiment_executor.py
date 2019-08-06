import numpy as np


######################################################################
from models.logistic_regression import ARDetectorByLogisticRegression
from models.random_forest import ARDetectorByRandomForest
from models.svm import ARDetectorBySVMWithRBF, ARDetectorBySVMWithLinear
from preprocess.data_representation_preparer import DataRepresentationPreparer

target_drugs = ['Isoniazid', 'Rifampicin', 'Ethambutol', 'Pyrazinamide', 'Streptomycin', 'Ofloxacin', 'Amikacin', 'Ciprofloxacin', 'Moxifloxacin', 'Capreomycin', 'Kanamycin']
label_tags = 'phenotype'
TRADITIONAL_ML_SCORING = 'accuracy'
######################################################################

class ExperimentExecutor:
    def __init__(self, models, data_representation='binary'):
        self.data_representation = data_representation

        models_arr = models.split(',')
        self.enable_svm_linear = False
        self.enable_svm_rbf = False
        self.enable_rf = False
        self.enable_dnn = False
        self.enable_lr = False

        for model in models_arr:
            if model == 'svm_linear':
                self.enable_svm_linear = True
            if model == 'svm_rbf':
                self.enable_svm_rbf = True
            if model == 'rf':
                self.enable_rf = True
            if model == 'dnn':
                self.enable_dnn = True
            if model == 'lr':
                self.enable_lr = True

    def conduct_all_experiments(self, results_directory, feature_selection):
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
            self.conduct_5x2cv_for_model(ar_detector, _, _)
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
            self.conduct_5x2cv_for_model(ar_detector, _, _)

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
            self.conduct_5x2cv_for_model(ar_detector, _, _)

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
            self.conduct_5x2cv_for_model(ar_detector, _, _)

    def conduct_5x2cv_for_model(self, ar_detector, raw_feature_matrix, raw_labels):
        features_base_directory = '/run/media/herkut/herkut/TB_genomes/features/features'

        # models = {'svm_linear': 'path_to_best_json',
        #            'svm_rbf': 'path_to_best_json',
        #            'lr': 'path_to_best_json',
        #            'rf':'path_to_best_json'}

        for i in range(0, 5):
            ########################################
            #                                      #
            # Train on split 1 and test on split 2 #
            #                                      #
            ########################################
            for j in range(len(target_drugs)):
                x, y = self.filter_out_nan(raw_feature_matrix, raw_labels[target_drugs[i]])

            tr_indexes = np.genfromtxt('/run/media/herkut/herkut/TB_genomes/features/features/5xcv2_f_test_' + str(j + 1) + '/' + target_drugs[j] + '_split1.csv',
                                       delimiter=' ',
                                       dtype=np.int32)

            te_indexes = np.genfromtxt('/run/media/herkut/herkut/TB_genomes/features/features/5xcv2_f_test_' + str(j + 1) + '/' + target_drugs[j] + '_split2.csv',
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

            ar_detector.train()

            ########################################
            #                                      #
            # Train on split 2 and test on split 1 #
            #                                      #
            ########################################
            for j in range(len(target_drugs)):
                x, y = self.filter_out_nan(raw_feature_matrix, raw_labels[target_drugs[i]])

            tr_indexes = np.genfromtxt('/run/media/herkut/herkut/TB_genomes/features/features/5xcv2_f_test_' + str(j + 1) + '/' + target_drugs[j] + '_split2.csv',
                                       delimiter=' ',
                                       dtype=np.int32)

            te_indexes = np.genfromtxt('/run/media/herkut/herkut/TB_genomes/features/features/5xcv2_f_test_' + str(j + 1) + '/' + target_drugs[j] + '_split1.csv',
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


