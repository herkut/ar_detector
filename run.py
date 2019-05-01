import os

from docopt import docopt

from models.ModelManager import ModelManager
from preprocess.feature_label_preparer import FeatureLabelPreparer
from preprocess.find_mutations_on_target_genes import FindMutationsOnTargetGenes
from utils.input_parser import InputParser


feature_selections = {'1.b': 'feature_matrix_09_without_unique_mutations'
                        , '1.a': 'feature_matrix_09_with_all_mutations'
                        , '2.b': 'feature_matrix_like_baseline_without_unique_mutations'
                        , '2.a': 'feature_matrix_like_baseline_with_all_mutations'}


def main():
    args = docopt("""
    Usage: 
        run.py find_mutations <target_base_directory> <target_directory_ids>
        run.py train_models <directory_containing_feature_matrices> <models> <directory_containing_results>

    Options:
        -h --help   : show this
    """)

    target_base_directory = args['<target_base_directory>']

    if args['find_mutations']:
        target_directories_str = args['<target_directory_ids>']

        target_directories = InputParser.parse_input_string(target_directories_str)

        FindMutationsOnTargetGenes.initialize(target_directories)

        for target_directory in target_directories:
            FindMutationsOnTargetGenes.find_mutations_on_target_genes(target_base_directory + str(target_directory) + '/')

        FindMutationsOnTargetGenes.save_all_mutations()

    elif args['train_models']:
        models = args['<models>']
        feature_matrices_directory = args['<directory_containing_feature_matrices>']
        results_directory = args['<directory_containing_results>']

        feature_matrices_files = []
        for file in os.listdir('/run/media/herkut/herkut/TB_genomes/ar_detection_dataset/new_approach/'):
            feature_matrices_files.append(file)
            features_tr, labels_tr, features_te, labels_te = FeatureLabelPreparer.separate_and_get_features_like_baseline('/run/media/herkut/herkut/TB_genomes/ar_detection_dataset/new_approach/' + file)

        """
        model_manager = ModelManager(models)
        for k, v in feature_selections.items():
            features_tr, labels_tr, features_te, labels_te = FeatureLabelPreparer.separate_and_get_features_like_baseline(feature_matrices_directory + v + '.csv')
            model_manager.train_and_test_models(results_directory, k, features_tr, labels_tr, features_te, labels_te)
        """


if __name__ == '__main__':
    main()
