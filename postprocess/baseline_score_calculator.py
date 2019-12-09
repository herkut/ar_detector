import os
from config import Config
from preprocess.feature_label_preparer import FeatureLabelPreparer
from utils.confusion_matrix_drawer import classification_report
import numpy as np
from utils.helper_functions import get_index_to_remove
import matplotlib.pyplot as plt


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
    ground_truth = raw_label_matrix.loc[te_indexes, drug]

    if label_assignment == 0:
        result = np.zeros(ground_truth.shape[0])

    elif label_assignment == 1:
        result = np.ones(ground_truth.shape[0])

    elif label_assignment == 2:
        result = np.random.choice([0, 1],
                                  size=ground_truth.shape[0],
                                  p=[counts[0] / np.sum(counts), counts[1] / np.sum(counts)])

    else:
        raise Exception('Unknown label assignment')
    report = classification_report(ground_truth.values, result)
    return report


def find_class_ditributions(drug):
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
    sizes = [counts[0], counts[1], nan_element_count]

    return sizes


if __name__ == '__main__':
    #raw = open('/home/herkut/Desktop/ar_detector/configurations/conf.yml')
    raw = open('/run/media/herkut/hdd-1/TB_genomes/ar_detector/configurations/conf.yml')
    Config.initialize_configurations(raw)
    for i in range(3):
        print("Label assignment: " + str(i))
        for td in Config.target_drugs:
            result = create_classification_results_for_dummy_classifier(td, label_assignment=i)
            print("Sensitivity: " + '{:.2f}'.format(result['sensitivity/recall'])
                  + " Specificity: " + '{:.2f}'.format(result['specificity'])
                  + " Precision: " + '{:.2f}'.format(result['precision'])
                  + " F1: " + '{:.2f}'.format(result['f1']))

    fig, ax = plt.subplots(2, 2)
    labels = ['Susceptible', 'Resistant', 'Not labeled']
    colors = ['gold', 'lightcoral', 'lightskyblue']

    inh_sizes = find_class_ditributions('Isoniazid')
    ax[0, 0].pie(inh_sizes, labels=labels, autopct='%.1f%%', colors=colors)
    ax[0, 0].set_title('Class distribution for Isoniazid')

    rif_sizes = find_class_ditributions('Rifampicin')
    ax[0, 1].pie(rif_sizes, labels=labels, autopct='%.1f%%', colors=colors)
    ax[0, 1].set_title('Class distribution for Rifampicin')

    emb_sizes = find_class_ditributions('Ethambutol')
    ax[1, 0].pie(emb_sizes, labels=labels, autopct='%.1f%%', colors=colors)
    ax[1, 0].set_title('Class distribution for Ethambutol')

    pza_sizes = find_class_ditributions('Pyrazinamide')
    ax[1, 1].pie(pza_sizes, labels=labels, autopct='%.1f%%', colors=colors)
    ax[1, 1].set_title('Class distribution for Pyrazinamide')

    fig.subplots_adjust(wspace=.2)
    plt.tight_layout()
    fig.savefig('/home/herkut/Desktop/class_distributions.png')
    """
    for td in Config.target_drugs:
        find_class_ditributions(td)
        fig, ax = plt.subplots()
        plt.title('Class distribution for ' + drug)
        ax.
        ax.set_aspect('equal')
    """

