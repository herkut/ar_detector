import os 
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


def filter_out_nan(x, y):
    index_to_remove = y[y.isna()].index

    for i in y.index:
        print(str(i) + '--' + str(i in x.index))
        print(str(i) + '--' + str(i in y.index))
        print()

    xx = x.drop(index_to_remove, inplace=False)
    yy = y.drop(index_to_remove, inplace=False)
    
    return xx, yy


def get_labels_from_file(file_containing_labels):
    labels = pd.read_csv(file_containing_labels, index_col=0, dtype={0: str})
    return labels


def get_feature_matrix_from_files(feature_files):
    raw_feature_matrix = pd.read_csv(feature_files[0], index_col=0, dtype={0: str})
    # print(raw_feature_matrix.shape)
    for i in range(1, len(feature_files)):
        tmp_feature_matrix = pd.read_csv(feature_files[i], index_col=0, dtype={0: str})
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


if __name__ == '__main__':
    dataset_directory = '/run/media/herkut/herkut/TB_genomes/ar_detection_dataset/'
    dataset = 'dataset-ii'
    data_representation = 'binary'
    target_drugs = ['Isoniazid', 'Rifampicin', 'Ethambutol', 'Pyrazinamide']

    for td in target_drugs:
        if dataset == 'dataset-i':
            raw_feature_selections = {'snp_09_bcf_nu_indel_00_platypus_all': [
                    os.path.join(dataset_directory, 'new_approach_with_normalization', 'snp_bcftools_0.9_notunique.csv'),
                    os.path.join(dataset_directory, 'new_approach_with_normalization', 'indel_platypus_0.0_all.csv')]
            }
            label_file = 'labels.csv'
        elif dataset == 'dataset-ii':
            raw_feature_selections = {'snp_09_bcf_nu_indel_00_platypus_all': [
                    os.path.join(dataset_directory, 'features_dataset_ii_with_normalization', 'snp_bcftools_0.9_notunique.csv'),
                    os.path.join(dataset_directory, 'features_dataset_ii_with_normalization', 'indel_platypus_0.0_all.csv')]
            }
            label_file = 'labels_dataset-ii.csv'
        ##
        feature_selections = {}
        for k, v in raw_feature_selections.items():
            feature_selections[data_representation + '_' + k] = v

        for td in target_drugs:
            for k, v in feature_selections.items():
                raw_label_matrix = get_labels_from_file(os.path.join(dataset_directory, label_file))

                raw_feature_matrix = get_feature_matrix_from_files(v)
                # raw_feature_matrix.to_csv(index=True)
                x, y = filter_out_nan(raw_feature_matrix, raw_label_matrix[td])

                sss = StratifiedShuffleSplit(test_size=0.2, random_state=0)

                for train_index, test_index in sss.split(x, y):
                    print("TRAIN:", train_index, "TEST:", test_index)
