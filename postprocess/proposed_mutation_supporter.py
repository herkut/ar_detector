import csv
import os
import pickle
import sys

import pandas as pd

from config import Config
from postprocess.postprocessor import PostProcessor
from postprocess.xgboost_feature_extractor import XGBoostFeatureExtractor
from preprocess.feature_label_preparer import FeatureLabelPreparer
from run import get_labels_and_raw_feature_selections


def convert_pandas_to_pickle():
    label_file, raw_feature_selections = get_labels_and_raw_feature_selections('dataset-ii')

    feature_selections = {}
    for k, v in raw_feature_selections.items():
        feature_selections['binary' + '_' + k] = v

    for k, v in feature_selections.items():
        print("Feature importance would be extacted for: " + k)
        raw_label_matrix = FeatureLabelPreparer.get_labels_from_file(os.path.join(Config.dataset_directory, label_file))
        raw_feature_matrix = FeatureLabelPreparer.get_feature_matrix_from_files(v)

        raw_label_matrix.to_pickle(
            os.path.join(Config.dataset_directory, 'features_dataset_ii_with_normalization', 'labels.pkl'))
        raw_feature_matrix.to_pickle(
            os.path.join(Config.dataset_directory, 'features_dataset_ii_with_normalization', k + '.pkl'))


def create_database_according_to_dreamtb(drug):
    pp = PostProcessor()

    directory = '/run/media/herkut/herkut/TB_genomes/ar_detection_dataset/mutation_database_related'
    prefix = 'DownloadDB'
    if drug == 'Isoniazid':
        file_name = prefix + '_INH.csv'
    elif drug == 'Rifampicin':
        file_name = prefix + '_RIF.csv'
    elif drug == 'Ethambutol':
        file_name = prefix + '_EMB.csv'
    elif drug == 'Pyrazinamide':
        file_name = prefix + '_PZA.csv'
    else:
        print('Unknown drug: ' + drug)
        sys.exit(1)

    with open(os.path.join(directory,
                           file_name)) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        counter = 0
        for row in readCSV:
            if counter > 1:
                mutation = pp.create_mutation_id_from_dreamtb_record(row)
                if mutation is None:
                    with open(os.path.join(directory,
                                           'manuel_operation_required_mutations_' + drug + '.csv'), 'a') as fd:
                        fd.write('Line ' + str(counter + 1) + ' none value' + "\n")
                else:
                    with open(os.path.join(directory,
                                           'mutation_database_' + drug + '.csv'), 'a') as fd:
                        fd.write(mutation + "\n")
            counter = counter + 1


class ProposedMutationSupporter:
    def __init__(self,
                 models_directory,
                 important_feature_count=15,
                 enable_most_important_feature_statistics_estimator=True,
                 enable_random_feature_statistics_estimator=True):
        self.models_directory = models_directory
        self.important_feature_count = important_feature_count
        self.enable_most_important_feature_statistics_estimator = enable_most_important_feature_statistics_estimator
        self.enable_random_feature_statistics_estimator = enable_random_feature_statistics_estimator
        self.thresholds = [10, 20, 40, 60]
        self.raw_labels = None
        self.raw_features = None
        self.mutations_labels_joined = None
        self.read_feature_and_labels()

        self.mutation_database = {}
        self.important_features = {}
        self.random_features = {}

        self.pp = PostProcessor()

        self.initialize_mutation_database()
        self.find_most_important_features()
        self.choose_random_features()

        field_names = ['mutation_id', 'resistant_count', 'susceptible_count', 'ratio']

        if self.enable_most_important_feature_statistics_estimator:
            for drug in self.important_features:
                all_statistics = []
                for mutation in self.important_features[drug][0]:
                    mutation_id = None
                    resistant = None
                    susceptible = None
                    ratio = None
                    if self.find_mutation_type(mutation) == 'snp' \
                            or self.find_mutation_type(mutation) == 'insertion' \
                            or self.find_mutation_type(mutation) == 'deletion':
                        mutation_id, resistant, susceptible, ratio = self.find_statistics_about_mutation(mutation, drug)
                        all_statistics.append({'mutation_id': mutation_id,
                                               'resistant_count': resistant,
                                               'susceptible_count': susceptible,
                                               'ratio': ratio})
                with open(os.path.join(Config.results_directory, 'mutations_' + drug + '.csv'), 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=field_names)
                    writer.writeheader()
                    writer.writerows(all_statistics)

        if self.enable_random_feature_statistics_estimator:
            for drug in self.random_features:
                all_statistics = []
                for mutation in self.random_features[drug][0]:
                    mutation_id = None
                    resistant = None
                    susceptible = None
                    ratio = None
                    if self.find_mutation_type(mutation) == 'snp' \
                            or self.find_mutation_type(mutation) == 'insertion' \
                            or self.find_mutation_type(mutation) == 'deletion':
                        mutation_id, resistant, susceptible, ratio = self.find_statistics_about_mutation(mutation, drug)
                        all_statistics.append({'mutation_id': mutation_id,
                                               'resistant_count': resistant,
                                               'susceptible_count': susceptible,
                                               'ratio': ratio})
                with open(os.path.join(Config.results_directory, 'random_mutations_' + drug + '.csv'), 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=field_names)
                    writer.writeheader()
                    writer.writerows(all_statistics)
        self.create_results()

    def initialize_mutation_database(self):
        for drug in Config.target_drugs:
            with open(os.path.join(Config.mutation_database_directory,
                                   'mutation_database_' + drug + '.csv'), 'r') as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            self.mutation_database[drug] = content

    def create_results(self):
        precisions = {}
        for drug in Config.target_drugs:
            print('For ' + drug)
            for t in self.thresholds:
                precisions[t] = self.calculate_precision(drug, t)
                print(str(t) + ': ' + str(precisions[t]))

    def calculate_precision(self, drug, threshold):
        counter = 0
        tp = 0
        fp = 0
        for i in range(len(self.important_features[drug][0])):
            if counter < threshold:
                if self.find_mutation_type(self.important_features[drug][0][i]) != 'mnv':
                    # print(self.pp.find_mutation_id(self.important_features[drug][0][i]))
                    if self.pp.find_mutation_id(self.important_features[drug][0][i]) in self.mutation_database[drug]:
                        tp += 1
                    else:
                        fp += 1
                    counter += 1
            else:
                break
        return tp / (tp + fp)

    def find_mutation_type(self, mutation):
        mutation_arr = mutation.split('_')
        if len(mutation_arr[1]) == len(mutation_arr[2]) == 1:
            return 'snp'
        elif len(mutation_arr[1]) > len(mutation_arr[2]):
            return 'deletion'
        elif len(mutation_arr[1]) < len(mutation_arr[2]):
            return 'insertion'
        else:
            return 'mnv'

    def choose_random_features(self):
        for drug in Config.target_drugs:
            m = XGBoostFeatureExtractor(os.path.join(self.models_directory,
                                                     'best_models',
                                                     'xgboost_accuracy_phenotype_binary_snp_09_bcf_nu_indel_00_platypus_all',
                                                     'xgboost' + '_' + drug + '.sav'),
                                        self.raw_features.columns,
                                        self.raw_features)
            self.random_features[drug] = m.choose_features_randomly(self.important_feature_count)

    def find_most_important_features(self):
        for drug in Config.target_drugs:
            m = XGBoostFeatureExtractor(os.path.join(self.models_directory,
                                                     'best_models',
                                                     'xgboost_accuracy_phenotype_binary_snp_09_bcf_nu_indel_00_platypus_all',
                                                     'xgboost' + '_' + drug + '.sav'),
                                        self.raw_features.columns,
                                        self.raw_features)
            self.important_features[drug] = m.find_most_important_n_features(self.important_feature_count)

    def find_statistics_about_mutation(self, mutation, drug):
        mutation_id = self.pp.find_mutation_id(mutation)
        results1 = self.mutations_labels_joined.loc[self.mutations_labels_joined[mutation] == 1].loc[self.mutations_labels_joined[drug] == 1].loc[:, [mutation, drug]]
        tp = results1.shape[0]
        results2 = self.mutations_labels_joined.loc[self.mutations_labels_joined[mutation] == 1].loc[self.mutations_labels_joined[drug] == 0].loc[:, [mutation, drug]]
        fp = results2.shape[0]
        return mutation_id, tp, fp, (tp/(tp+fp))

    def read_feature_and_labels(self):
        label_file, raw_feature_selections = get_labels_and_raw_feature_selections('dataset-ii')

        feature_selections = {}
        for k, v in raw_feature_selections.items():
            feature_selections['binary' + '_' + k] = v

        for k, v in feature_selections.items():
            self.raw_labels = pd.read_pickle(os.path.join(Config.dataset_directory, 'features_dataset_ii_with_normalization', 'labels.pkl'))
            self.raw_features = pd.read_pickle(os.path.join(Config.dataset_directory, 'features_dataset_ii_with_normalization', k + '.pkl'))

        self.mutations_labels_joined = self.raw_labels.join(self.raw_features)


if __name__ == '__main__':
    ar_detector_directory = '/run/media/herkut/hdd-1/TB_genomes/ar_detector'
    # ar_detector_directory = '/home/herkut/Desktop/ar_detector'
    raw = open(os.path.join(ar_detector_directory,
                            'configurations/conf.yml'))
    Config.initialize_configurations(raw)

    models_directory = '/run/media/herkut/herkut/TB_genomes/truba/ar_detector_results_dataset-ii_20191205'
    # models_directory = '/run/media/herkut/hdd-1/TB_genomes/truba/ar_detector_results_dataset-ii_20191205'

    # Create python database for dreamtb downloaded mutations
    for drug in Config.target_drugs:
        create_database_according_to_dreamtb(drug)

    # convert_pandas_to_pickle()
    pms = ProposedMutationSupporter(models_directory,
                                    important_feature_count=150,
                                    enable_most_important_feature_statistics_estimator=False,
                                    enable_random_feature_statistics_estimator=False)
