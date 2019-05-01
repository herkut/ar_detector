import os

from docopt import docopt

from models.model_manager import ModelManager
from preprocess.feature_label_preparer import FeatureLabelPreparer
from preprocess.find_mutations_on_target_genes import FindMutationsOnTargetGenes
from utils.input_parser import InputParser

"""
feature_selections = {'1.b': 'feature_matrix_09_without_unique_mutations'
                        , '1.a': 'feature_matrix_09_with_all_mutations'
                        , '2.b': 'feature_matrix_like_baseline_without_unique_mutations'
                        , '2.a': 'feature_matrix_like_baseline_with_all_mutations'}
"""


def main():
    args = docopt("""
    Usage: 
        run.py find_mutations <target_base_directory> <target_directory_ids>
        run.py train_models <models> <directory_containing_results>

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
        results_directory = args['<directory_containing_results>']

        feature_selections = {'snp_09_bcf_nu_indel_00_platypus_all': ['/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_bcftools_0.9_notunique.csv', '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_platypus_0.0_all.csv'],
                              'snp_09_bcf_nu_indel_09_bcf_all': ['/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_bcftools_0.9_notunique.csv', '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_bcftools_0.0_all.csv'],
                              'snp_09_playpus_nu_indel_00_platypus_all': ['/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_platypus_0.9_notunique.csv', '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_platypus_0.0_all.csv'],
                              'snp_09_playpus_nu_indel_00_bcf_all': ['/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_platypus_0.9_notunique.csv', '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_bcftools_0.0_all.csv']
                              }

        model_manager = ModelManager(models)
        for k, v in feature_selections.items():
            raw_feature_matrix = FeatureLabelPreparer.get_feature_matrix_from_files(v)
            raw_label_matrix = FeatureLabelPreparer.get_labels_from_file('/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/labels.csv')
            model_manager.train_and_test_models(results_directory, k, raw_feature_matrix, raw_label_matrix)


if __name__ == '__main__':
    main()
