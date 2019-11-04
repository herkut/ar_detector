import os

from docopt import docopt

from models.cnn_model_manager import CNNModelManager
from models.model_manager import ModelManager
from preprocess.feature_label_preparer import FeatureLabelPreparer
from utils.helper_functions import get_index_to_remove
from utils.statistical_tests.experiment_executor import ExperimentExecutor
from config import Config
import numpy as np


def get_labels_and_raw_feature_selections(dataset):
    # As Arzucan Ozgur suggested, we focus on the feature selection approach in the reference paper,
    # please check old_raw_feature_selection file for alternatives
    if dataset == 'dataset-i':
        raw_feature_selections = {'snp_09_bcf_nu_indel_00_platypus_all': [
            os.path.join(Config.dataset_directory,
                         'new_approach_with_normalization',
                         'snp_bcftools_0.9_notunique.csv'),
            os.path.join(Config.dataset_directory,
                         'new_approach_with_normalization',
                         'indel_platypus_0.0_all.csv')]
        }
        label_file = 'labels.csv'
    elif dataset == 'dataset-ii':
        raw_feature_selections = {'snp_09_bcf_nu_indel_00_platypus_all': [
            os.path.join(Config.dataset_directory,
                         'features_dataset_ii_with_normalization',
                         'sorted_snp_bcftools_0.9_notunique.csv'),
            os.path.join(Config.dataset_directory,
                         'features_dataset_ii_with_normalization',
                         'sorted_indel_platypus_0.0_all.csv')]
        }
        label_file = 'sorted_labels_dataset-ii.csv'
    else:
        raise Exception('Unknown dataset: ' + dataset)

    return label_file, raw_feature_selections


def main():
    args = docopt("""
    Usage: 
        run.py tune_hyperparameters <configuration_file> <models> [--data_representation=<data_representation>]
        run.py train_best_models <configuration_file> <models> [--data_representation=<data_representation>]
        run.py test_best_models <configuration_file> <models> [--data_representation=<data_representation>]
        run.py cnn_tune_hyperparameters <configuration_file> <models>
        run.py cnn_train_best_models <configuration_file> <models>
        run.py cnn_test_best_models <configuration_file> <models>
        run.py execute_experiments <configuration_file> <models> [--data_representation=<data_representation>]
        run.py select_best_model <configuration_file> <directory_containing_results>
        
    Options:
        -h --help   : show this
        --data_representation=<data_representation> which data representation would be used: [tfidf|tfrf|bm25tfidf|bm25tfrf]
    """)

    data_representation = 'binary'
    configuration_file = args['<configuration_file>']

    raw = open(configuration_file)
    Config.initialize_configurations(raw)
    dataset = Config.target_dataset

    if args['--data_representation']:
        data_representation = args['--data_representation']

    if args['tune_hyperparameters']:
        models = args['<models>']
        results_directory = args['<directory_containing_results>']

        label_file, raw_feature_selections = get_labels_and_raw_feature_selections(dataset)

        feature_selections = {}
        for k, v in raw_feature_selections.items():
            feature_selections[data_representation + '_' + k] = v

        model_manager = ModelManager(models, dataset, data_representation=data_representation)
        for k, v in feature_selections.items():
            print("Models would be trained and tested for feature selection method: " + k)
            raw_label_matrix = FeatureLabelPreparer.get_labels_from_file(os.path.join(Config.dataset_directory, label_file))
            raw_feature_matrix = FeatureLabelPreparer.get_feature_matrix_from_files(v)
            model_manager.tune_train_and_test_models(k, raw_feature_matrix, raw_label_matrix)

    elif args['test_best_models']:
        models = args['<models>']
        results_directory = args['<directory_containing_results>']

        label_file, raw_feature_selections = get_labels_and_raw_feature_selections(dataset)

        feature_selections = {}
        for k, v in raw_feature_selections.items():
            feature_selections[data_representation + '_' + k] = v

        model_manager = ModelManager(models, dataset, data_representation=data_representation)
        for k, v in feature_selections.items():
            print("Models would be trained and tested for feature selection method: " + k)
            raw_label_matrix = FeatureLabelPreparer.get_labels_from_file(os.path.join(Config.dataset_directory, label_file))
            raw_feature_matrix = FeatureLabelPreparer.get_feature_matrix_from_files(v)
            model_manager.test_best_models(k, raw_feature_matrix, raw_label_matrix)

    elif args['train_best_models']:
        models = args['<models>']
        results_directory = args['<directory_containing_results>']

        label_file, raw_feature_selections = get_labels_and_raw_feature_selections(dataset)

        feature_selections = {}
        for k, v in raw_feature_selections.items():
            feature_selections[data_representation + '_' + k] = v

        model_manager = ModelManager(models, dataset, data_representation=data_representation)
        for k, v in feature_selections.items():
            print("Models would be trained and tested for feature selection method: " + k)
            raw_label_matrix = FeatureLabelPreparer.get_labels_from_file(os.path.join(Config.dataset_directory, label_file))
            raw_feature_matrix = FeatureLabelPreparer.get_feature_matrix_from_files(v)
            model_manager.train_and_test_best_models(k, raw_feature_matrix, raw_label_matrix)

    elif args['cnn_tune_hyperparameters']:
        models = args['<models>']
        # TODO use cnn model manager
        cnn_model_manager = CNNModelManager(models, dataset)

        cnn_model_manager.tune_train_and_test_models()

    elif args['cnn_train_best_models']:
        models = args['<models>']
        # TODO use cnn model manager
        cnn_model_manager = CNNModelManager(models, dataset)

        cnn_model_manager.train_and_test_best_models()

    elif args['cnn_test_best_models']:
        models = args['<models>']
        # TODO use cnn model manager
        cnn_model_manager = CNNModelManager(models, dataset)

        cnn_model_manager.test_best_models()

    elif args['execute_experiments']:
        models = args['<models>']
        results_directory = args['<directory_containing_results>']

        label_file, raw_feature_selections = get_labels_and_raw_feature_selections(dataset)

        feature_selections = {}
        for k, v in raw_feature_selections.items():
            feature_selections[data_representation + '_' + k] = v

        for k, v in feature_selections.items():
            print("Model results would be prepared for 5x2cv paired f test for: " + k)
            raw_label_matrix = FeatureLabelPreparer.get_labels_from_file(os.path.join(Config.dataset_directory, label_file))
            raw_feature_matrix = FeatureLabelPreparer.get_feature_matrix_from_files(v)
            experiment_executor = ExperimentExecutor(models,
                                                     data_representation=data_representation)

            experiment_executor.conduct_all_experiments(Config.results_directory,
                                                        k,
                                                        data_representation,
                                                        raw_feature_matrix,
                                                        raw_label_matrix)


if __name__ == '__main__':
    main()
