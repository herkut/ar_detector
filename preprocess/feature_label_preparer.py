import pandas as pd
import numpy as np
from sklearn.externals import joblib

from models.random_forest import ARDetectorByRandomForest
from models.svm_rbf import ARDetectorBySVMWithRBF

FEATURE_MATRIX_DIRECTORY = '/run/media/herkut/herkut/TB_genomes/ar_detection_dataset/'
#FEATURE_MATRIX_FILE = 'feature_matrix_09_with_all_mutations.csv'
FEATURE_MATRIX_FILE = 'feature_matrix_09_without_unique_mutations.csv'

LABELS_DIRECTORY = '/run/media/herkut/herkut/TB_genomes/ar_detection_dataset/'
LABELS_FILE = 'labels.csv'

antibiotic_names_phenotype = ['Isoniazid', 'Rifampicin', 'Ethambutol', 'Pyrazinamide', 'Streptomycin', 'Ciprofloxacin', 'Moxifloxacin', 'Ofloxacin', 'Amikacin', 'Capreomycin', 'Kanamycin']
antibiotic_names_genotype = ['Isoniazid.1', 'Rifampicin.1', 'Ethambutol.1', 'Pyrazinamide.1', 'Streptomycin.1', 'Ciprofloxacin.1', 'Moxifloxacin.1', 'Ofloxacin.1', 'Amikacin.1', 'Capreomycin.1', 'Kanamycin.1']

target_drugs = ['Isoniazid', 'Rifampicin', 'Ethambutol', 'Pyrazinamide', 'Streptomycin', 'Ofloxacin', 'Amikacin']

#label_tags = 'genotype'
label_tags = 'phenotype'


def extract_labels_from_excel():
    if label_tags == 'phenotype':
        labels = pd.read_excel('/run/media/herkut/herkut/TB_genomes/baseline/mmc2.xlsx', sheet_name='All phenotypes and genotypes', usecols=antibiotic_names_phenotype, skiprows=2)
    elif label_tags == 'genotype':
        labels = pd.read_excel('/run/media/herkut/herkut/TB_genomes/baseline/mmc2.xlsx', sheet_name='All phenotypes and genotypes', usecols=antibiotic_names_genotype, skiprows=2)
    else:
        print('Unknown label type')
        exit(1)

    labels.index += 1

    for index, row in labels.iterrows():
        for j, column in row.iteritems():
            if labels.at[index, j] == 'S':
                labels.at[index, j] = 0
            elif labels.at[index, j] == 'R':
                labels.at[index, j] = 1
            elif labels.at[index, j] == 'No result':
                labels.at[index, j] = np.nan

    labels.to_csv(r'' + LABELS_DIRECTORY + LABELS_FILE, index=True, header=True)


def filter_mutations_occured_only_once(features):
    removed_mutations = []
    for column in features:
        x = features[column].value_counts()
        if x[1] <= 1:
            removed_mutations.append(column)
            features.drop(column, 1, inplace=True)
    return features, removed_mutations


def filter_out_empty_rows(features):
    # filter rows with no valid data in them (0 for all mutations)
    index_would_be_used = []
    index_would_be_ignored = []

    for index, row in features.iterrows():
        if np.sum(row[:]) > 0:
            index_would_be_used.append(index)
        else:
            index_would_be_ignored.append(index)

    return index_would_be_used, index_would_be_ignored


def filter_out_nan(x, y):
    index_to_remove = np.argwhere(np.isnan(y))

    xx = np.delete(x, index_to_remove, axis=0)
    yy = np.delete(y, index_to_remove, axis=0)

    return xx, yy


def train_svm_with_rbf(ar_detector, index_of_antibiotic, feature_matrix_training, labels_matrix_training, feature_matrix_test, labels_matrix_test):
    # conduct svm model
    c_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # c_range = [1]
    gamma_range = [0.001, 0.1, 1, 10, 100]
    # gamma_range = [1]
    x_tr, y_tr = filter_out_nan(feature_matrix_training, labels_matrix_training[:, index_of_antibiotic])
    x_te, y_te = filter_out_nan(feature_matrix_test, labels_matrix_test[:, index_of_antibiotic])

    print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
    print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))
    print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

    ar_detector.initialize_datasets(x_tr, y_tr, x_te, y_te)

    ar_detector.tune_hyperparameters(c_range, gamma_range)

    print(ar_detector._best_model)


def train_random_forest(ar_detector, index_of_antibiotic, feature_matrix_training, labels_matrix_training, feature_matrix_test, labels_matrix_test):
    #bootstrap = [True, False]
    n_estimators = [int(x) for x in np.linspace(start=100, stop=500, num=100)]
    max_features = ['sqrt', 'log2', None]

    x_tr, y_tr = filter_out_nan(feature_matrix_training, labels_matrix_training[:, index_of_antibiotic])
    x_te, y_te = filter_out_nan(feature_matrix_test, labels_matrix_test[:, index_of_antibiotic])

    print('For ' + ar_detector._antibiotic_name + ' feature and label sizes')
    print('Training ' + str(x_tr.shape) + ' ' + str(y_tr.shape))
    print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

    ar_detector.initialize_datasets(x_tr, y_tr, x_te, y_te)

    ar_detector.tune_hyperparameters(n_estimators, max_features)

    print(ar_detector._best_model)


def test_svm_with_rbf(ar_detector, index_of_antibiotic, feature_matrix_training, labels_matrix_training, feature_matrix_test, labels_matrix_test):
    x_tr, y_tr = filter_out_nan(feature_matrix_training, labels_matrix_training[:, index_of_antibiotic])
    x_te, y_te = filter_out_nan(feature_matrix_test, labels_matrix_test[:, index_of_antibiotic])

    ar_detector.initialize_datasets(x_tr, y_tr, x_te, y_te)
    ar_detector.load_model()
    ar_detector.test_model()


def test_random_forest(ar_detector, index_of_antibiotic, feature_matrix_training, labels_matrix_training, feature_matrix_test, labels_matrix_test):
    x_tr, y_tr = filter_out_nan(feature_matrix_training, labels_matrix_training[:, index_of_antibiotic])
    x_te, y_te = filter_out_nan(feature_matrix_test, labels_matrix_test[:, index_of_antibiotic])

    ar_detector.initialize_datasets(x_tr, y_tr, x_te, y_te)
    ar_detector.load_model()
    ar_detector.test_model()


def main():
    raw_feature_matrix = pd.read_csv(FEATURE_MATRIX_DIRECTORY + FEATURE_MATRIX_FILE, index_col=0)

    raw_feature_matrix, removed_mutations = filter_mutations_occured_only_once(raw_feature_matrix)

    labels = pd.read_csv(LABELS_DIRECTORY + LABELS_FILE, index_col=0)

    #index_would_be_used, index_would_be_ignored = filter_out_empty_rows(raw_feature_matrix)
    index_would_be_used = []

    for i in range(1, 3652):
        index_would_be_used.append(i)

    index_training = filter(lambda x: x < 2100, index_would_be_used)

    index_test = filter(lambda x: x >= 2100, index_would_be_used)

    # finds training features for isolate that would be investigated
    features_training = raw_feature_matrix.loc[index_training, :]

    feature_matrix_training = features_training.values

    # finds test features for isolate that would be investigated
    features_test = raw_feature_matrix.loc[index_test, :]

    feature_matrix_test = features_test.values

    # finds training labels for isolate that would be investigated
    labels_training = labels.loc[index_training, :]

    labels_matrix_training = labels_training.values

    # finds test labels for isolate that would be investigated
    labels_test = labels.loc[index_test, :]

    labels_matrix_test = labels_test.values

    #####################################
    #                                   #
    #           SVM with rbf            #
    #                                   #
    #####################################

    for i in range(len(target_drugs)):
        ar_detector = ARDetectorBySVMWithRBF(target_drugs[i], label_tags=label_tags, scoring='f1')
        # train the model
        train_svm_with_rbf(ar_detector, i, feature_matrix_training, labels_matrix_training, feature_matrix_test, labels_matrix_test)
        # test the model
        ar_detector = ARDetectorBySVMWithRBF(target_drugs[i], label_tags=label_tags, scoring='f1')
        test_svm_with_rbf(ar_detector, i, feature_matrix_training, labels_matrix_training, feature_matrix_test, labels_matrix_test)

    """
    #####################################
    #                                   #
    #           Random Forest           #
    #                                   #
    #####################################

    for i in range(4):
        ar_detector = ARDetectorByRandomForest(antibiotic_names_phenotype[i], label_tags=label_tags, scoring='f1')
        # train the model
        train_random_forest(ar_detector, i, feature_matrix_training, labels_matrix_training, feature_matrix_test, labels_matrix_test)
        # test the model
        ar_detector = ARDetectorByRandomForest(antibiotic_names_phenotype[i], label_tags=label_tags, scoring='f1')
        test_svm_with_rbf(ar_detector, i, feature_matrix_training, labels_matrix_training, feature_matrix_test, labels_matrix_test)
    """

    #####################################
    #                                   #
    #       Deep Neural Network         #
    #                                   #
    #####################################


if __name__ == '__main__':
    main()
