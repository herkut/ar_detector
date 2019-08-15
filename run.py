import os

from docopt import docopt

from models.model_manager import ModelManager
from models.tensorflow_models.tensorflow_model_manager import TensorflowModelManager
from preprocess.feature_label_preparer import FeatureLabelPreparer
from preprocess.find_mutations_on_target_genes import FindMutationsOnTargetGenes
from utils.input_parser import InputParser
from utils.statistical_tests.experiment_executor import ExperimentExecutor


def main():
    args = docopt("""
    Usage: 
        run.py find_mutations <target_base_directory> <target_directory_ids>
        run.py train_models <models> <directory_containing_results> [--data_representation=<data_representation>]
        run.py train_tensorflow_models <models> <directory_containing_results> [--data_representation=<data_representation>]
        run.py execute_experiments <models> <directory_containing_results> [--data_representation=<data_representation>]
        run.py select_best_model <directory_containing_results>
        
    Options:
        -h --help   : show this
        --data_representation=<data_representation> which data representation would be used: [tfidf|tfrf|bm25tfidf|bm25tfrf]
    """)

    target_base_directory = args['<target_base_directory>']

    data_representation = 'binary'

    if args['--data_representation']:
        data_representation = args['--data_representation']

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
        """
        raw_feature_selections = {'snp_09_bcf_nu_indel_00_platypus_all': [
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_bcftools_0.9_notunique.csv',
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_platypus_0.0_all.csv'],
                              'snp_09_bcf_nu_indel_09_bcf_all': [
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_bcftools_0.9_notunique.csv',
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_bcftools_0.0_all.csv'],
                              'snp_09_platiypus_nu_indel_00_platypus_all': [
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_platypus_0.9_notunique.csv',
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_platypus_0.0_all.csv'],
                              'snp_09_platypus_nu_indel_00_bcf_all': [
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_platypus_0.9_notunique.csv',
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_bcftools_0.0_all.csv'],
                              'snp_09_bcf_platypus_nu_indel_00_bcf_platypus_all': [
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_bcftools_0.9_notunique.csv',
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_platypus_0.9_notunique.csv',
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_bcftools_0.0_all.csv',
                                  '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_platypus_0.0_all.csv']
                              }
        """
        # As Arzucan Özgür suggested, we focus on the feature selection approach in the reference paper
        raw_feature_selections = {'snp_09_bcf_nu_indel_00_platypus_all': [
                                    '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_bcftools_0.9_notunique.csv',
                                    '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_platypus_0.0_all.csv']
        }

        feature_selections = {}
        for k, v in raw_feature_selections.items():
            feature_selections[data_representation + '_' + k] = v

        model_manager = ModelManager(models, data_representation=data_representation)
        for k, v in feature_selections.items():
            print("Models would be trained and tested for feature selection method: " + k)
            raw_label_matrix = FeatureLabelPreparer.get_labels_from_file('/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/labels.csv')
            raw_feature_matrix = FeatureLabelPreparer.get_feature_matrix_from_files(v)
            model_manager.train_and_test_models(results_directory, k, raw_feature_matrix, raw_label_matrix)

    elif args['train_tensorflow_models']:
        models = args['<models>']
        results_directory = args['<directory_containing_results>']
        """
        raw_feature_selections = {'snp_09_bcf_nu_indel_00_platypus_all': [
                                    '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_bcftools_0.9_notunique.csv',
                                    '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_platypus_0.0_all.csv'],
                                  'snp_09_bcf_nu_indel_09_bcf_all': [
                                    '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_bcftools_0.9_notunique.csv',
                                    '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_bcftools_0.0_all.csv'],
                                  'snp_09_platiypus_nu_indel_00_platypus_all': [
                                    '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_platypus_0.9_notunique.csv',
                                    '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_platypus_0.0_all.csv'],
                                  'snp_09_platypus_nu_indel_00_bcf_all': [
                                    '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_platypus_0.9_notunique.csv',
                                    '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_bcftools_0.0_all.csv'],
                                  'snp_09_bcf_platypus_nu_indel_00_bcf_platypus_all': [
                                    '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_bcftools_0.9_notunique.csv',
                                    '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_platypus_0.9_notunique.csv',
                                    '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_bcftools_0.0_all.csv',
                                    '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_platypus_0.0_all.csv']
                                  }
        """
        # As Arzucan Özgür suggested, we focus on the feature selection approach in the reference paper
        raw_feature_selections = {'snp_09_bcf_nu_indel_00_platypus_all': [
            '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_bcftools_0.9_notunique.csv',
            '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_platypus_0.0_all.csv']
        }

        feature_selections = {}
        for k, v in raw_feature_selections.items():
            feature_selections[data_representation + '_' + k] = v

        model_manager = TensorflowModelManager(models, results_directory, data_representation=data_representation)
        for k, v in feature_selections.items():
            print("Models would be trained and tested for feature selection method: " + k)
            model_manager.set_feature_selection(k)
            raw_label_matrix = FeatureLabelPreparer.get_labels_from_file('/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/labels.csv')
            raw_feature_matrix = FeatureLabelPreparer.get_feature_matrix_from_files(v)
            model_manager.train_and_test_models(raw_feature_matrix, raw_label_matrix)

    elif args['execute_experiments']:
        models = args['<models>']
        results_directory = args['<directory_containing_results>']

        # As Arzucan Özgür suggested, we focus on the feature selection approach in the reference paper
        raw_feature_selections = {'snp_09_bcf_nu_indel_00_platypus_all': [
            '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/snp_bcftools_0.9_notunique.csv',
            '/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/new_approach/indel_platypus_0.0_all.csv']
        }

        feature_selections = {}
        for k, v in raw_feature_selections.items():
            feature_selections[data_representation + '_' + k] = v

        for k, v in feature_selections.items():
            print("Model results would be prepared for 5x2cv paired f test for: " + k)
            raw_label_matrix = FeatureLabelPreparer.get_labels_from_file('/run/media/herkut/hdd-1/TB_genomes/ar_detection_dataset/labels.csv')
            raw_feature_matrix = FeatureLabelPreparer.get_feature_matrix_from_files(v)
            experiment_executor = ExperimentExecutor(models,
                                                     data_representation=data_representation)

            experiment_executor.conduct_all_experiments('/run/media/herkut/hdd-1/TB_genomes/ar_detector_results/',
                                                        k,
                                                        data_representation,
                                                        raw_feature_matrix,
                                                        raw_label_matrix)


if __name__ == '__main__':
    main()
