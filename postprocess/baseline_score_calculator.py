import os

from config import Config
from preprocess.feature_label_preparer import FeatureLabelPreparer
from utils.confusion_matrix_drawer import classification_report
import numpy as np

from utils.helper_functions import get_index_to_remove


def create_classification_results_for_dummy_classifier(drug, label_assignment=0):
    """

    :param label_assignment: 0 -> assign all classification results to 0, 1-> assign all classification results to 1,
    2-> assign classfication results randomly with respect to its resistant susceptible distribution
    :return: return dummy classfiers results
    """
    raw_label_matrix = FeatureLabelPreparer.get_labels_from_file(os.path.join(Config.dataset_directory,
                                                                              'sorted_labels_dataset-ii.csv'))
    isolate_counts = raw_label_matrix.shape[0]
    non_existing = []
    predefined_file_to_remove = ['8316-09', 'NL041']

    index_to_remove = get_index_to_remove(raw_label_matrix[drug])

    for ne in predefined_file_to_remove:
        if ne not in index_to_remove:
            non_existing.append(ne)

    raw_label_matrix.drop(index_to_remove, inplace=True)
    raw_label_matrix.drop(non_existing, inplace=True)

    tr_indexes = np.genfromtxt(os.path.join(Config.dataset_index_directory + '_' + Config.target_dataset,
                                            drug + '_tr_indices.csv'),
                               delimiter=' ',
                               dtype=str)

    te_indexes = np.genfromtxt(os.path.join(Config.dataset_index_directory + '_' + Config.target_dataset,
                                            drug + '_te_indices.csv'),
                               delimiter=' ',
                               dtype=str)

    unique, counts = np.unique(raw_label_matrix[drug].values, return_counts=True)

    nan_element_count = isolate_counts - np.sum(counts)
    ground_truth = raw_label_matrix[te_indexes]

    if label_assignment == 0:

        pass
    elif label_assignment == 1:

        pass
    elif label_assignment == 2:

        pass
    else:
        raise Exception('Unknown label assignment')


if __name__ == '__main__':
    raw = open('/home/herkut/Desktop/ar_detector/configurations/conf.yml')
    Config.initialize_configurations(raw)

    for td in Config.target_drugs:
        create_classification_results_for_dummy_classifier(td)
