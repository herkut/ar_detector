import os

import progressbar
import torch
from torch.utils.data.dataset import Dataset
import numpy as np

from config import Config
from preprocess.feature_label_preparer import FeatureLabelPreparer
from utils.helper_functions import get_index_to_remove


class CNNDataset(Dataset):

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
    def create_sequences(cls, raw_labels, target_drug):
        cls.ordered_genes = ['gyrB', 'gyrA', 'iniA', 'iniC', 'rpoB', 'rpsL', 'embR', 'rrs', 'fabG1', 'inhA', 'rpsA',
                             'tlyA', 'ndh', 'katG', 'pncA', 'eis', 'ahpC', 'manB', 'rmlD', 'embC', 'embA', 'embB',
                             'gidB']

        if not os.path.exists(Config.cnn_dataset_directory):
            os.makedirs(Config.cnn_dataset_directory)

        if not os.path.exists(os.path.join(Config.cnn_dataset_directory, target_drug)):
            os.makedirs(os.path.join(Config.cnn_dataset_directory, target_drug))

        result_directory = os.path.join(Config.cnn_dataset_directory, target_drug)

        non_existing = []
        predefined_file_to_remove = ['8316-09', 'NL041']

        index_to_remove = get_index_to_remove(raw_labels[target_drug])

        for ne in predefined_file_to_remove:
            if ne not in index_to_remove:
                non_existing.append(ne)

        raw_labels.drop(index_to_remove, inplace=True)
        raw_labels.drop(non_existing, inplace=True)

        idx = raw_labels.index
        labels = raw_labels[target_drug].values

        sequences = {}

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
                    np.concatenate((tmp_sequence, gene_int))
                f.close()
            sequences[i] = torch.nn.functional.one_hot(torch.from_numpy(tmp_sequence).long(), 5)

            np.savez(os.path.join(result_directory, i + '.npz'), sequences[i].numpy())

            z += 1
            pbar.update(z)
        pbar.finish()

    def __init__(self, raw_labels, target_drug):
        self.ordered_genes = ['gyrB', 'gyrA', 'iniA', 'iniC', 'rpoB', 'rpsL', 'embR', 'rrs', 'fabG1', 'inhA', 'rpsA',
                              'tlyA', 'ndh', 'katG', 'pncA', 'eis', 'ahpC', 'manB', 'rmlD', 'embC', 'embA', 'embB',
                              'gidB']
        self.target_drug = target_drug
        self.sequences = {}

        if not os.path.exists(Config.cnn_dataset_directory):
            os.makedirs(Config.cnn_dataset_directory)

        if not os.path.exists(os.path.join(Config.cnn_dataset_directory, self.target_drug)):
            os.makedirs(os.path.join(Config.cnn_dataset_directory, self.target_drug))

        self.sequences_directory = os.path.join(Config.cnn_dataset_directory, self.target_drug)

        non_existing = []
        predefined_file_to_remove = ['8316-09', 'NL041']

        index_to_remove = get_index_to_remove(raw_labels[target_drug])

        for ne in predefined_file_to_remove:
            if ne not in index_to_remove:
                non_existing.append(ne)

        raw_labels.drop(index_to_remove, inplace=True)
        raw_labels.drop(non_existing, inplace=True)

        self.idx = raw_labels.index
        self.labels = raw_labels[target_drug].values

        self.read_sequences_from_files()

    def read_sequences_from_files(self):
        for i in self.idx.values:
            tmp_sequence = np.load(os.path.join(self.sequences_directory, i + '.npz'))
            self.sequences[i] = torch.from_numpy(tmp_sequence)

    def __getitem__(self, index):
        return self.sequences[self.idx[index]]

    def __len__(self):
        return len(self.sequences)


if __name__ == '__main__':
    configuration_file = '/home/herkut/Desktop/ar_detector/configurations/conf.yml'
    raw = open(configuration_file)
    Config.initialize_configurations(raw)

    raw_label_matrix = FeatureLabelPreparer.get_labels_from_file(os.path.join(Config.dataset_directory,
                                                                              'labels_dataset-ii.csv'))

    for td in Config.target_drugs:
        CNNDataset.create_sequences(raw_label_matrix, td)

    """
    dataset = CNNDataset(raw_label_matrix, Config.target_drugs[3])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

    for i, data in enumerate(dataloader, 0):
        print(i)
        if i == 177:
            print(data)
    """