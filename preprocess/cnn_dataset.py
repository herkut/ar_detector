import os
import timeit

import progressbar
import torch
from torch.utils.data.dataset import Dataset
import numpy as np

from config import Config
from preprocess.feature_label_preparer import FeatureLabelPreparer
from utils.helper_functions import get_index_to_remove, get_k_fold


class ARCNNDataset(Dataset):
    def __init__(self, idx, labels, target_drug):
        self.ordered_genes = ['gyrB', 'gyrA', 'iniA', 'iniC', 'rpoB', 'rpsL', 'embR', 'rrs', 'fabG1', 'inhA', 'rpsA',
                              'tlyA', 'ndh', 'katG', 'pncA', 'eis', 'ahpC', 'manB', 'rmlD', 'embC', 'embA', 'embB',
                              'gidB']
        self.target_drug = target_drug
        self.sequences_directory = os.path.join(Config.cnn_dataset_directory)

        self.idx = idx
        self.labels = np.reshape(labels, (-1, 1))

    def __getitem__(self, index):
        tmp_sequence = np.load(os.path.join(self.sequences_directory, self.idx[index] + '.npz'))
        sequence = torch.from_numpy(tmp_sequence['arr_0'])
        labels = torch.from_numpy(self.labels[index]).long()

        return (sequence, labels)

    def __len__(self):
        return labels.shape[0]


class CNNDataset(Dataset):
    whole_sequences = None

    @classmethod
    def read_sequences_from_files(cls, sequences_directory, idx):
        cls.whole_sequences = {}
        for i in idx.values:
            tmp_sequence = np.load(os.path.join(sequences_directory, i + '.npz'))
            cls.whole_sequences[i] = torch.from_numpy(tmp_sequence['arr_0'])

    @classmethod
    def _convert_str_to_int(cls, sequence):
        # A -> 0 -> 1 0 0 0 0
        # T -> 1 -> 0 1 0 0 0
        # G -> 2 -> 0 0 1 0 0
        # C -> 3 -> 0 0 0 1 0
        # - -> 4 -> 0 0 0 0 1
        res = np.zeros(len(sequence))
        char2int = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '-': 4}
        for i in range(len(sequence)):
            res[i] = char2int[sequence[i]]
        return res

    @classmethod
    def create_and_store_sequences(cls, raw_labels):
        cls.ordered_genes = ['gyrB', 'gyrA', 'iniA', 'iniC', 'rpoB', 'rpsL', 'embR', 'rrs', 'fabG1', 'inhA', 'rpsA',
                             'tlyA', 'ndh', 'katG', 'pncA', 'eis', 'ahpC', 'manB', 'rmlD', 'embC', 'embA', 'embB',
                             'gidB']

        if not os.path.exists(Config.cnn_dataset_directory):
            os.makedirs(Config.cnn_dataset_directory)

        result_directory = os.path.join(Config.cnn_dataset_directory)

        non_existing = []
        predefined_file_to_remove = ['8316-09', 'NL041']
        """
        index_to_remove = get_index_to_remove(raw_labels[target_drug])

        for ne in predefined_file_to_remove:
            if ne not in index_to_remove:
                non_existing.append(ne)

        raw_labels.drop(index_to_remove, inplace=True)
        """
        raw_labels.drop(predefined_file_to_remove, inplace=True)

        idx = raw_labels.index

        # sequences = {}

        progressbar_size = len(idx)
        pbar = progressbar.ProgressBar(maxval=progressbar_size)
        pbar.start()
        z = 0
        for i in idx.values:
            tmp_sequence = None
            for gene in cls.ordered_genes:
                mutated_gene_file = os.path.join(Config.base_directory, i, 'mutated_genes', gene + '_expanded')
                f = open(mutated_gene_file, 'r')
                gene = f.read()
                gene_int = CNNDataset._convert_str_to_int(gene)
                if tmp_sequence is None:
                    tmp_sequence = gene_int
                else:
                    tmp_sequence = np.concatenate((tmp_sequence, gene_int))
                f.close()
            sequence = torch.nn.functional.one_hot(torch.from_numpy(tmp_sequence).long(), 5)

            np.savez(os.path.join(result_directory, i + '.npz'), sequence)

            z += 1
            pbar.update(z)
        pbar.finish()

    def __init__(self, idx, labels, target_drug):
        self.ordered_genes = ['gyrB', 'gyrA', 'iniA', 'iniC', 'rpoB', 'rpsL', 'embR', 'rrs', 'fabG1', 'inhA', 'rpsA',
                              'tlyA', 'ndh', 'katG', 'pncA', 'eis', 'ahpC', 'manB', 'rmlD', 'embC', 'embA', 'embB',
                              'gidB']
        self.target_drug = target_drug
        self.sequences_directory = os.path.join(Config.cnn_dataset_directory, self.target_drug)
        self.sequences = {}

        self.idx = idx
        self.labels = labels
        self.sequences = {k: CNNDataset.whole_sequences[k] for k in idx}

    def __getitem__(self, index):
        return self.sequences[self.idx[index]]

    def __len__(self):
        return len(self.sequences)


def create_and_store_sequences():
    raw_label_matrix = FeatureLabelPreparer.get_labels_from_file(os.path.join(Config.dataset_directory,
                                                                              'labels_dataset-ii.csv'))
    CNNDataset.create_and_store_sequences(raw_label_matrix)


def test_stored_sequences():
    raw_label_matrix = FeatureLabelPreparer.get_labels_from_file(os.path.join(Config.dataset_directory,
                                                                              'sorted_labels_dataset-ii.csv'))
    target_drug = Config.target_drugs[0]

    if not os.path.exists(Config.cnn_dataset_directory):
        os.makedirs(Config.cnn_dataset_directory)

    if not os.path.exists(os.path.join(Config.cnn_dataset_directory, target_drug)):
        os.makedirs(os.path.join(Config.cnn_dataset_directory, target_drug))

    sequences_directory = Config.cnn_dataset_directory

    non_existing = []
    predefined_file_to_remove = ['8316-09', 'NL041']

    index_to_remove = get_index_to_remove(raw_label_matrix[target_drug])

    for ne in predefined_file_to_remove:
        if ne not in index_to_remove:
            non_existing.append(ne)

    raw_label_matrix.drop(index_to_remove, inplace=True)
    raw_label_matrix.drop(non_existing, inplace=True)

    idx = raw_label_matrix.index
    labels = raw_label_matrix[target_drug].values

    CNNDataset.read_sequences_from_files(sequences_directory, idx)

    cv = get_k_fold(10)

    for train_index, test_index in cv.split(idx, labels):
        tr_dataset = CNNDataset(idx[train_index], labels[train_index], target_drug)
        tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=64)
        te_dataset = CNNDataset(idx[test_index], labels[test_index], target_drug)
        te_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=64)


def find_bactera_with_all_labels():
    raw_label_matrix = FeatureLabelPreparer.get_labels_from_file(os.path.join(Config.dataset_directory,
                                                                              'sorted_labels_dataset-ii.csv'))
    for i in Config.target_drugs:
        target_drug = i

        non_existing = []
        predefined_file_to_remove = ['8316-09', 'NL041']

        index_to_remove = get_index_to_remove(raw_label_matrix[target_drug])

        for ne in predefined_file_to_remove:
            if ne not in index_to_remove and ne in raw_label_matrix.index:
                non_existing.append(ne)

        raw_label_matrix.drop(index_to_remove, inplace=True)
        raw_label_matrix.drop(non_existing, inplace=True)
    return raw_label_matrix


if __name__ == '__main__':
    configuration_file = '/home/herkut/Desktop/ar_detector/configurations/conf.yml'
    raw = open(configuration_file)
    Config.initialize_configurations(raw)

    # create_and_store_sequences()

    # test_stored_sequences()
    rlm = find_bactera_with_all_labels()

    raw_label_matrix = FeatureLabelPreparer.get_labels_from_file(os.path.join(Config.dataset_directory,
                                                                              'sorted_labels_dataset-ii.csv'))

    target_drug = Config.target_drugs[0]

    non_existing = []
    predefined_file_to_remove = ['8316-09', 'NL041']

    index_to_remove = get_index_to_remove(raw_label_matrix[target_drug])

    for ne in predefined_file_to_remove:
        if ne not in index_to_remove:
            non_existing.append(ne)

    raw_label_matrix.drop(index_to_remove, inplace=True)
    raw_label_matrix.drop(non_existing, inplace=True)

    idx = raw_label_matrix.index
    labels = raw_label_matrix[target_drug].values

    cv = get_k_fold(10)

    for train_index, test_index in cv.split(idx, labels):
        tr_dataset = ARCNNDataset(idx[train_index], labels[train_index], target_drug)
        tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=64)
        te_dataset = ARCNNDataset(idx[test_index], labels[test_index], target_drug)
        te_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=64)

        start = timeit.default_timer()
        for i_batch, sample_batched in enumerate(tr_dataloader):
            print(i_batch)
        stop = timeit.default_timer()
        print('Execution time for iterating whole dataset: ', stop-start)

    print('Zaa')
