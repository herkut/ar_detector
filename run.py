import os

from docopt import docopt

from models.model_manager import ModelManager
from preprocess.feature_label_preparer import FeatureLabelPreparer
from preprocess.find_mutations_on_target_genes import FindMutationsOnTargetGenes
from utils.input_parser import InputParser


def main():
    args = docopt("""
    Usage: 
        run.py find_mutations <target_base_directory> <target_directory_ids>
        run.py train_models <models> <directory_containing_results> [--tfidf=<tfidf>]

    Options:
        -h --help   : show this
        --tdidf=<tdidf> whether tdidf would be used or not
    """)

    target_base_directory = args['<target_base_directory>']

    use_tfidf = False

    if args['--tfidf']:
        print('Feature matrix would be created with tf-idf')
        use_tfidf = args['--tfidf']

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

        feature_selections = {'snp_09_bcf_nu_indel_00_platypus_all': [
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_bcftools_0.9_notunique.csv',
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_platypus_0.0_all.csv'],
                              'snp_09_bcf_nu_indel_09_bcf_all': [
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_bcftools_0.9_notunique.csv',
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_bcftools_0.0_all.csv'],
                              'snp_09_playpus_nu_indel_00_platypus_all': [
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_platypus_0.9_notunique.csv',
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_platypus_0.0_all.csv'],
                              'snp_09_playpus_nu_indel_00_bcf_all': [
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_platypus_0.9_notunique.csv',
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_bcftools_0.0_all.csv'],
                              'snp_09_bcf_platypus_nu_indel_00_bcf_platypus_all': [
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_bcftools_0.9_notunique.csv',
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_platypus_0.9_notunique.csv',
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_bcftools_0.0_all.csv',
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_platypus_0.0_all.csv']
                              }

        if use_tfidf:
            for k, v in feature_selections.items():
                feature_selections['tfidf_' + k] = v
                del feature_selections[k]

        model_manager = ModelManager(models)
        for k, v in feature_selections.items():
            print("Models would be trained and tested for feature selection method: " + k)
            raw_feature_matrix = FeatureLabelPreparer.get_feature_matrix_from_files(v, use_tfidf=use_tfidf)
            raw_label_matrix = FeatureLabelPreparer.get_labels_from_file('/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/labels.csv')
            model_manager.train_and_test_models(results_directory, k, raw_feature_matrix, raw_label_matrix)


if __name__ == '__main__':
    main()
