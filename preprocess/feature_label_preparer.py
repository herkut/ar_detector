import pandas as pd
import numpy as np


######################################################################################
BASE_DIRECTORY='/run/media/herkut/herkut/TB_genomes/'

FEATURE_MATRIX_DIRECTORY = BASE_DIRECTORY + 'ar_detection_dataset/'

FEATURE_MATRIX_FILE_PREFIX = 'feature_matrix_'

LABELS_DIRECTORY = BASE_DIRECTORY + 'ar_detection_dataset/'
LABELS_FILE = 'labels.csv'

antibiotic_names_phenotype = ['Isoniazid', 'Rifampicin', 'Ethambutol', 'Pyrazinamide', 'Streptomycin', 'Ciprofloxacin', 'Moxifloxacin', 'Ofloxacin', 'Amikacin', 'Capreomycin', 'Kanamycin']
antibiotic_names_genotype = ['Isoniazid.1', 'Rifampicin.1', 'Ethambutol.1', 'Pyrazinamide.1', 'Streptomycin.1', 'Ciprofloxacin.1', 'Moxifloxacin.1', 'Ofloxacin.1', 'Amikacin.1', 'Capreomycin.1', 'Kanamycin.1']

#label_tags = 'genotype'
label_tags = 'phenotype'

#######################################################################################
IGNORE_EMPTY_ROWS = False
ENABLE_SVM = True
ENABLE_RF = False
ENABLE_DNN = False
######################################################################################


class FeatureLabelPreparer:
    @staticmethod
    def extract_labels_from_excel():
        if label_tags == 'phenotype':
            labels = pd.read_excel('/run/media/herkut/herkut/TB_genomes/baseline/mmc2.xlsx',
                                   sheet_name='All phenotypes and genotypes', usecols=antibiotic_names_phenotype,
                                   skiprows=2)
        elif label_tags == 'genotype':
            labels = pd.read_excel('/run/media/herkut/herkut/TB_genomes/baseline/mmc2.xlsx',
                                   sheet_name='All phenotypes and genotypes', usecols=antibiotic_names_genotype,
                                   skiprows=2)
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

    @staticmethod
    def filter_mutations_occurred_only_once(features):
        removed_mutations = []
        filtered_mutations = None
        for column in features:
            x = features[column].value_counts()
            if x[1] <= 1:
                removed_mutations.append(column)
                filtered_mutations = features.drop(column, 1, inplace=False)
        return filtered_mutations, removed_mutations

    @staticmethod
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

    @staticmethod
    def separate_and_get_features_like_baseline(feature_matrix_file):
        raw_feature_matrix = pd.read_csv(feature_matrix_file, index_col=0)

        filtered_feature_matrix = raw_feature_matrix
        # filtered_feature_matrix, removed_mutations = filter_mutations_occurred_only_once(raw_feature_matrix)

        labels = pd.read_csv(LABELS_DIRECTORY + LABELS_FILE, index_col=0)

        if IGNORE_EMPTY_ROWS:
            index_would_be_used, index_would_be_ignored = FeatureLabelPreparer.filter_out_empty_rows(
                filtered_feature_matrix)
        else:
            index_would_be_used = []

            for i in range(1, 3652):
                index_would_be_used.append(i)

        index_training = filter(lambda x: x < 2100, index_would_be_used)

        index_test = filter(lambda x: x >= 2100, index_would_be_used)

        # finds training features for isolate that would be investigated
        features_training = filtered_feature_matrix.loc[index_training, :]

        feature_matrix_training = features_training.values

        # finds test features for isolate that would be investigated
        features_test = filtered_feature_matrix.loc[index_test, :]

        feature_matrix_test = features_test.values

        if IGNORE_EMPTY_ROWS:
            index_would_be_used, index_would_be_ignored = FeatureLabelPreparer.filter_out_empty_rows(
                filtered_feature_matrix)
        else:
            index_would_be_used = []

            for i in range(1, 3652):
                index_would_be_used.append(i)

        index_training = filter(lambda x: x < 2100, index_would_be_used)

        index_test = filter(lambda x: x >= 2100, index_would_be_used)

        # finds training labels for isolate that would be investigated
        labels_training = labels.loc[index_training, :]

        labels_matrix_training = labels_training.values

        # finds test labels for isolate that would be investigated
        labels_test = labels.loc[index_test, :]

        labels_matrix_test = labels_test.values

        return feature_matrix_training, labels_matrix_training, feature_matrix_test, labels_matrix_test

    @staticmethod
    def get_feature_matrix_from_files(feature_files):
        raw_feature_matrix = pd.read_csv(feature_files[0], index_col=0, dtype={0: str})
        raw_feature_matrix.index = raw_feature_matrix.index.astype(str)
        # print(raw_feature_matrix.shape)
        for i in range(1, len(feature_files)):
            tmp_feature_matrix = pd.read_csv(feature_files[i], index_col=0, dtype={0: str})
            tmp_feature_matrix.index = tmp_feature_matrix.index.astype(str)
            # print(tmp_feature_matrix.shape)
            for column in tmp_feature_matrix.columns:
                if column in raw_feature_matrix.columns:
                    for j in raw_feature_matrix.index:
                        if raw_feature_matrix.at[j, column] == 0 and tmp_feature_matrix.at[j, column] == 1:
                            raw_feature_matrix.at[j, column] = 1
                else:
                    raw_feature_matrix[column] = tmp_feature_matrix[column]

        # print(raw_feature_matrix.shape)
        return raw_feature_matrix

    @staticmethod
    def get_labels_from_file(file_containing_labels):
        labels = pd.read_csv(file_containing_labels, index_col=0, dtype={0: str})
        labels.index = labels.index.astype(str)
        return labels


def main():
    tmp_arr = ['/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_bcftools_0.9_notunique.csv', '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_platypus_0.0_all.csv']
    raw_matrix = FeatureLabelPreparer.get_feature_matrix_from_files(tmp_arr, use_tfidf=True)
    labels = FeatureLabelPreparer.get_labels_from_file('/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/labels.csv')

    print('Zaa')


if __name__ == '__main__':
    main()
