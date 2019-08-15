import json
import numpy as np

######################################################################
from models.logistic_regression import ARDetectorByLogisticRegression
from models.random_forest import ARDetectorByRandomForest
from models.svm import ARDetectorBySVMWithRBF, ARDetectorBySVMWithLinear
from preprocess.data_representation_preparer import DataRepresentationPreparer
from utils.confusion_matrix_drawer import classification_report
from utils.helper_functions import conduct_data_preprocessing
from utils.numpy_encoder import NumpyEncoder

target_drugs = ['Isoniazid', 'Rifampicin', 'Ethambutol', 'Pyrazinamide', 'Streptomycin', 'Ofloxacin', 'Amikacin', 'Ciprofloxacin', 'Moxifloxacin', 'Capreomycin', 'Kanamycin']
label_tags = 'phenotype'
TRADITIONAL_ML_SCORING = 'accuracy'
directory_containing_best_model_informations = '/run/media/herkut/herkut/TB_genomes/ar_detector_results/best_models/'
results_directory_5x2cv_paired_f_test = '/run/media/herkut/herkut/TB_genomes/ar_detector_results/5x2cv_f_tests/'
######################################################################


class ExperimentExecutor:
    def __init__(self, models, data_representation='binary'):
        self.data_representation = data_representation

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

    def conduct_all_experiments(self, results_directory, feature_selection, data_representation, raw_feature_matrix, raw_labels):
        #####################################
        #                                   #
        #           SVM with rbf            #
        #                                   #
        #####################################
        if self.enable_svm_rbf:
            ar_detector = ARDetectorBySVMWithRBF(results_directory,
                                                 feature_selection,
                                                 label_tags=label_tags,
                                                 scoring=TRADITIONAL_ML_SCORING)
            self.conduct_5x2cv_for_model(ar_detector, 'svm_rbf', raw_feature_matrix, raw_labels, data_representation=data_representation)
        #####################################
        #                                   #
        #         SVM with linear           #
        #                                   #
        #####################################
        if self.enable_svm_linear:
            ar_detector = ARDetectorBySVMWithLinear(results_directory,
                                                    feature_selection,
                                                    label_tags=label_tags,
                                                    scoring=TRADITIONAL_ML_SCORING)
            self.conduct_5x2cv_for_model(ar_detector, 'svm_linear', raw_feature_matrix, raw_labels, data_representation=data_representation)

        #####################################
        #                                   #
        #           Random Forest           #
        #                                   #
        #####################################
        if self.enable_rf:
            ar_detector = ARDetectorByRandomForest(results_directory,
                                                   feature_selection,
                                                   label_tags=label_tags,
                                                   scoring=TRADITIONAL_ML_SCORING)
            self.conduct_5x2cv_for_model(ar_detector, 'rf', raw_feature_matrix, raw_labels, data_representation=data_representation)

        #####################################
        #                                   #
        #        Logistic Regression        #
        #                                   #
        #####################################
        if self.enable_lr:
            ar_detector = ARDetectorByLogisticRegression(results_directory,
                                                         feature_selection,
                                                         label_tags=label_tags,
                                                         scoring=TRADITIONAL_ML_SCORING)
            self.conduct_5x2cv_for_model(ar_detector, 'lr', raw_feature_matrix, raw_labels, data_representation=data_representation)

    def conduct_5x2cv_for_model(self, ar_detector, model, raw_feature_matrix, raw_labels, data_representation=None):
        # model may be svm_linear, svm_rbf, rf, lr as string
        features_base_directory = '/run/media/herkut/herkut/TB_genomes/features/features/'
        results = {}

        for j in range(len(target_drugs)):
            results[target_drugs[j]] = []
            for i in range(0, 5):
                # Data preprocessing
                iteration_results = []
                x, y = conduct_data_preprocessing(raw_feature_matrix, raw_labels[target_drugs[i]], data_representation)

                x_mat = x.values
                y_mat = y.values

                class_weights_arr = {}

                unique, counts = np.unique(y_mat, return_counts=True)

                class_weights = {0: counts[1] / (counts[0] + counts[1]), 1: counts[0] / (counts[0] + counts[1])}

                ar_detector.set_antibiotic_name(target_drugs[j])

                # load best parameters and reinitialize the model with these parameters
                with open('/run/media/herkut/herkut/TB_genomes/ar_detector_results/best_models/' + ar_detector._target_directory + model + '_' + target_drugs[j] + '.json') as json_data:
                    parameters = json.load(json_data)

                ########################################
                #                                      #
                # Train on split 1 and test on split 2 #
                #                                      #
                ########################################
                tr_indexes = np.genfromtxt(features_base_directory + '5xcv2_f_test_' + str(i + 1) + '/' + target_drugs[j] + '_split1.csv',
                                           delimiter=' ',
                                           dtype=np.int32)

                te_indexes = np.genfromtxt(features_base_directory + '5xcv2_f_test_' + str(i + 1) + '/' + target_drugs[j] + '_split2.csv',
                                           delimiter=' ',
                                           dtype=np.int32)

                # Update weights of features if necessary
                x_train = x.loc[tr_indexes].values
                y_train = y.loc[tr_indexes].values
                x_test = x.loc[te_indexes].values
                y_test = y.loc[te_indexes].values

                ar_detector.reinitialize_model_with_parameters(parameters, class_weights=class_weights)

                ar_detector.train(x_train, y_train)

                y_pred = ar_detector.predict(x_test)

                iteration_results.append(classification_report(y_test, y_pred))

                ########################################
                #                                      #
                # Train on split 2 and test on split 1 #
                #                                      #
                ########################################
                tr_indexes = np.genfromtxt(features_base_directory + '5xcv2_f_test_' + str(i + 1) + '/' + target_drugs[j] + '_split2.csv',
                                           delimiter=' ',
                                           dtype=np.int32)

                te_indexes = np.genfromtxt(features_base_directory + '5xcv2_f_test_' + str(i + 1) + '/' + target_drugs[j] + '_split1.csv',
                                           delimiter=' ',
                                           dtype=np.int32)

                # Update weights of features if necessary
                x_train = x.loc[tr_indexes].values
                y_train = y.loc[tr_indexes].values
                x_test = x.loc[te_indexes].values
                y_test = y.loc[te_indexes].values

                # load best parameters and reinitialize the model with these parameters
                with open('/run/media/herkut/herkut/TB_genomes/ar_detector_results/best_models/' + ar_detector._target_directory + model + '_' + target_drugs[j] + '.json') as json_data:
                    parameters = json.load(json_data)

                ar_detector.reinitialize_model_with_parameters(parameters, class_weights=class_weights)

                ar_detector.train(x_train, y_train)

                y_pred = ar_detector.predict(x_test)

                classification_report(y_test, y_pred)

                iteration_results.append(classification_report(y_test, y_pred))

            results[target_drugs[j]].append(iteration_results)
        # TODO print results dictionary as json into a file to conduct statistical tests for models later
        with open(results_directory_5x2cv_paired_f_test + target_drugs[j] + '/' + model + '.json', 'w') as f:
            f.write(json.dumps(results, cls=NumpyEncoder))
