import os

from config import Config
from postprocess.lr_feature_extractor import LogisticRegressionFeatureExtractor
from postprocess.rf_feature_extractor import RandomForestFeatureExtractor
from postprocess.xgboost_feature_extractor import XGBoostFeatureExtractor
from preprocess.feature_label_preparer import FeatureLabelPreparer
from run import get_labels_and_raw_feature_selections
from Bio.Seq import Seq
from Bio import SeqIO
import math
import json
import numpy as np
import matplotlib.pyplot as plt


def save_most_important_features(features, file):
    with open(file, 'w') as file:
        file.write(json.dumps(features, cls=NumpyEncoder))  # use `json.loads` to do the reverse


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class PostProcessor:
    target_genes = ['ahpC', 'eis', 'embA', 'embB', 'embC', 'embR', 'fabG1', 'gidB', 'gyrA', 'gyrB', 'inhA', 'iniA',
                    'iniC', 'katG', 'manB', 'ndh', 'pncA', 'rmlD', 'rpoB', 'rpsA', 'rpsL', 'rrs', 'tlyA']
    aminoacid_abbreviations = {'Phe': 'F',
                               'Leu': 'L',
                               'Ile': 'I',
                               'Met': 'M',
                               'Val': 'V',
                               'Ser': 'S',
                               'Pro': 'P',
                               'Thr': 'T',
                               'Ala': 'A',
                               'Tyr': 'Y',
                               'His': 'H',
                               'Gln': 'G',
                               'Asn': 'N',
                               'Lys': 'K',
                               'Asp': 'D',
                               'Glu': 'E',
                               'Cys': 'C',
                               'Trp': 'W',
                               'Arg': 'R',
                               'Gly': 'G',
                               'STOP': 'END'}

    codon_to_aminoacid = {'ATG': 'START',
                          'TAA': 'END', 'TGA': 'END', 'TAG': 'END',
                          'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
                          'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
                          'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
                          'AAA': 'K', 'AAG': 'K',
                          'AAT': 'N', 'AAC': 'N',
                          'ATG': 'M',
                          'GAT': 'D', 'GAC': 'D',
                          'TTT': 'F', 'TTC': 'F',
                          'TGT': 'C', 'TGC': 'C',
                          'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
                          'CAA': 'Q', 'CAG': 'Q',
                          'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
                          'GAA': 'E', 'GAG': 'E',
                          'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
                          'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
                          'TGG': 'W',
                          'CAT': 'H', 'CAC': 'H',
                          'TAT': 'Y', 'TAC': 'Y',
                          'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
                          'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V'}

    target_antibiotic_genes = {'Isoniazid': ['ahpC',
                                             'fagB1',
                                             'inhA',
                                             'katG',
                                             'ndh'],
                               'Rifampicin': ['rpoB'],
                               'Ethambutol': ['embA',
                                              'embB',
                                              'embC',
                                              'embR',
                                              'iniA',
                                              'iniC',
                                              'manB',
                                              'rmlD'],
                               'Pyrazinamide': ['pncA',
                                                'rpsA']}
    target_genes_start_end_positions = None
    genes = None
    reference_genome = None

    def create_mutation_id_from_dreamtb_record(self, row):
        # 10th field if - it is in promoter, it > 0 in encoding region
        mutation_id = None
        gene = row[4]
        if gene == 'mabA':
            gene = 'fabG1'
        if row[10] is None or row[10] == '?' or row[10] == '' or row[10] == 'See Note':
            return None
        elif '-' in row[10] and not row[10].startswith('-'):
            if row[11].startswith('ins') or row[11].startswith('del'):
                location = row[10]
                if location == '':
                    return None
                else:
                    mutation_id = gene + '_' + (location if '-' not in location else location.split('-')[0]) + '_' + row[11].replace(' ', '')
            else:
                codon = row[13]
                if row[14] is not None and row[14] != '':
                    aa_from = self.aminoacid_abbreviations[row[14].split('/')[0].strip()]
                    aa_to = self.aminoacid_abbreviations[row[14].split('/')[1].strip()]
                else:
                    return None
                mutation_id = gene + '_' + aa_from + str(codon) + aa_to
        elif int(row[10]) < 0:
            if row[11].startswith('ins') or row[11].startswith('del'):
                if row[11].startswith('ins') or row[11].startswith('del'):
                    location = row[10]
                    if location == '':
                        return None
                    else:
                        mutation_id = gene + '_' + location + '_' + row[11].replace(' ', '')
            else:
                location = row[10]
                nucleotide_from = row[11].split('/')[0].strip()
                nucleotide_to = row[11].split('/')[1].strip()
                mutation_id = gene + '_' + nucleotide_from + str(location) + nucleotide_to
        elif int(row[10]) >= 0:
            if row[11].startswith('ins') or row[11].startswith('del'):
                if row[11].startswith('ins') or row[11].startswith('del'):
                    location = row[10]
                    if location == '':
                        return None
                    else:
                        mutation_id = gene + '_' + (location if '-' not in location else location.split('-')[0]) + '_' + row[11].replace(' ', '')
            else:
                codon = row[13]
                if row[14] is not None and row[14] != '':
                    aa_from = self.aminoacid_abbreviations[row[14].split('/')[0].strip()]
                    aa_to = self.aminoacid_abbreviations[row[14].split('/')[1].strip()]
                else:
                    return None
                mutation_id = gene + '_' + aa_from + str(codon) + aa_to
        else:
            print(row[10])
            return None
        return mutation_id

    def create_codons_for_all_genes(self):
        PostProcessor.genes = {}
        for gene_name in PostProcessor.target_genes:
            gene = {}
            codons, aminoacids, gene_complement, sequences, sequences_on_first_helix = self.create_codons_for_gene(gene_name)
            gene['codons'] = codons
            gene['aminoacids'] = aminoacids
            gene['gene_complement'] = gene_complement
            gene['sequence'] = sequences
            gene['sequence_on_first_helix'] = sequences_on_first_helix
            PostProcessor.genes[gene_name] = gene

    def create_codons_for_gene(self, gene, additional_base_pair_upstream=99):
        codons = []
        aminoacids = []
        sequences_on_first_helix = None

        gene_complement = False
        with open(os.path.join(Config.target_genes_directory, gene)) as f:
            first_line = f.readline()
            elements = first_line.split(' ')
            tmp = elements[0].replace('>', '').split(':')[1]
            tmp_arr = tmp.split('-')
            if tmp_arr[0].startswith('c'):
                gene_complement = True

            gene_str = ''
            while True:
                line = f.readline().strip()
                gene_str += line
                if not line:
                    break

            gene_str = Seq(gene_str)

            counter = 0
            tmp_codon = ''
            for nucleotide in gene_str:
                if counter > 2:
                    counter = 0
                    codons.append(tmp_codon)
                    aminoacids.append(PostProcessor.codon_to_aminoacid[tmp_codon])
                    tmp_codon = nucleotide
                else:
                    tmp_codon += nucleotide
                counter += 1

            if gene_complement:
                sequences_on_first_helix = gene_str.reverse_complement()
            else:
                sequences_on_first_helix = gene_str

        return codons, aminoacids, gene_complement, gene_str, sequences_on_first_helix

    def load_start_and_end_positions_for_all_target_genes(self, additional_base_pair_upstream=0):
        if PostProcessor.target_genes_start_end_positions is None:
            PostProcessor.target_genes_start_end_positions = {}
            for gene in PostProcessor.target_genes:
                with open(os.path.join(Config.target_genes_directory, gene)) as f:
                    first_line = f.readline()
                    elements = first_line.split(' ')
                    tmp = elements[0].replace('>', '').replace('c', '')
                    tmp_arr = []
                    for e in tmp.split(':')[1].split('-'):
                        tmp_arr.append(int(e))
                    min_el = min(tmp_arr)
                    max_el = max(tmp_arr)
                    tmp_arr[0] = min_el
                    tmp_arr[1] = max_el

                    if (int(tmp_arr[0]) - additional_base_pair_upstream) < 0:
                        tmp_arr[0] = str(0)
                    else:
                        tmp_arr[0] = str(int(tmp_arr[0]) - additional_base_pair_upstream)
                    PostProcessor.target_genes_start_end_positions[gene] = {'start': int(tmp_arr[0]),
                                                                            'end': int(tmp_arr[1])}

    def get_mutation_location_from_mutation_key(self, mutation_key):
        return int(mutation_key.split('_')[0])

    def find_mutated_gene_from_mutation_key(self, mutation_key, additional_base_pair_upstream=100):
        mutation_location = self.get_mutation_location_from_mutation_key(mutation_key)

        mutated_gene = None
        for gene in PostProcessor.target_genes_start_end_positions:
            if PostProcessor.target_genes_start_end_positions[gene]['start']-additional_base_pair_upstream <= mutation_location <= PostProcessor.target_genes_start_end_positions[gene]['end']:
                mutated_gene = gene
                break

        return mutated_gene

    def __init__(self):
        self.reference_genome = list(SeqIO.parse('/run/media/herkut/herkut/TB_genomes/reference_genome/mtb_h37rv_v3.fasta', 'fasta'))[0]._seq
        # self.reference_genome = list(SeqIO.parse('/run/media/herkut/hdd-1/TB_genomes/reference_genome/mtb_h37rv_v3.fasta', 'fasta'))[0]._seq
        self.load_start_and_end_positions_for_all_target_genes()
        self.create_codons_for_all_genes()

    def find_important_mutations(self, most_importance_features):
        important_mutations = []
        for mif in most_importance_features:
            mutation_name = ''
            mutated_gene = pp.find_mutated_gene_from_mutation_key(mif[0])
            location = int(mif[0].split('_')[0])
            mutation_from = mif[0].split('_')[1]
            mutation_to = mif[0].split('_')[2]

            print(str(location) + ': ' + mutation_from + ' -> ' + mutation_to)
            if len(mutation_from) == 1 and len(mutation_to) == 1:
                mutation_type = 'snp'
            else:
                if len(mutation_from) > len(mutation_to):
                    mutation_type = 'del'
                elif len(mutation_from) < len(mutation_to):
                    mutation_type = 'in'
                else:
                    mutation_type = 'mnv'

            if mutation_type == 'snp':
                location_on_helix_1 = location_on_gene = location - pp.target_genes_start_end_positions[mutated_gene]['start']

                if location_on_helix_1 >= 0:  # mutations on genes
                    if pp.genes[mutated_gene]['gene_complement']:
                        mutation_to = Seq(mutation_to).complement()
                        gene_length = pp.target_genes_start_end_positions[mutated_gene]['end'] - \
                                      pp.target_genes_start_end_positions[mutated_gene]['start']
                        location_on_gene = gene_length - location_on_helix_1
                    else:
                        location_on_gene = location_on_helix_1

                    codon_number = int(location_on_gene / 3)
                    codon_location = location_on_gene % 3
                    codon_from = pp.genes[mutated_gene]['codons'][codon_number]
                    codon_to = codon_from
                    codon_to = codon_to[:codon_location] + mutation_to + codon_to[codon_location + 1:]

                    mutation_name = mutated_gene + '_' + pp.codon_to_aminoacid[codon_from] + str(codon_number + 1) + \
                                    pp.codon_to_aminoacid[codon_to]

                    # print(mutation_name, str(mif[1]))
                    important_mutations.append({'mutation': mutation_name, 'score': mif[1]})
                else:  # mutations on gene promoters
                    if pp.genes[mutated_gene]['gene_complement']:
                        mutation_from = Seq(mutation_from).reverse_complement()._data
                        mutation_to = Seq(mutation_to).reverse_complement()._data
                    mutation_name = mutated_gene + '_' + mutation_from + str(location_on_gene) + mutation_to
                    # print(mutation_name, str(mif[1]))
                    important_mutations.append({'mutation': mutation_name, 'score': mif[1]})
            elif mutation_type == 'in':
                location_on_helix_1 = location_on_gene = location - pp.target_genes_start_end_positions[mutated_gene][
                    'start']

                if location_on_helix_1 >= 0:  # mutations on genes
                    if pp.genes[mutated_gene]['gene_complement']:
                        gene_length = pp.target_genes_start_end_positions[mutated_gene]['end'] - \
                                      pp.target_genes_start_end_positions[mutated_gene]['start']
                        location_on_gene = gene_length - location_on_helix_1
                    else:
                        location_on_gene = location_on_helix_1

                if pp.genes[mutated_gene]['gene_complement']:
                    mutation_from = Seq(mutation_from).reverse_complement()._data
                    mutation_to = Seq(mutation_to).reverse_complement()._data
                mutation_name = mutated_gene + '_' + str(location_on_gene) + '_in' + mutation_to[1:]
                # print(mutation_name, str(mif[1]))
                important_mutations.append({'mutation': mutation_name, 'score': mif[1]})
            elif mutation_type == 'del':
                if pp.genes[mutated_gene]['gene_complement']:
                    gene_length = pp.target_genes_start_end_positions[mutated_gene]['end'] - \
                                  pp.target_genes_start_end_positions[mutated_gene]['start']
                    location_on_gene = gene_length - location_on_helix_1
                else:
                    location_on_gene = location_on_helix_1

                if pp.genes[mutated_gene]['gene_complement']:
                    mutation_from = Seq(mutation_from).reverse_complement()._data
                    mutation_to = Seq(mutation_to).reverse_complement()._data
                mutation_name = mutated_gene + '_' + str(location_on_gene) + '_ins' + mutation_from[1:]
                # print(mutation_name, str(mif[1]))
                important_mutations.append({'mutation': mutation_name, 'score': mif[1]})
            else:
                print(mutation_from + ' -> ' + mutation_to)

        return important_mutations

    def find_mutation_id(self, mif):
        mutation_name = None
        mutated_gene = self.find_mutated_gene_from_mutation_key(mif)
        location = int(mif.split('_')[0])
        mutation_from = mif.split('_')[1]
        mutation_to = mif.split('_')[2]

        # print('Mutation on: ' + mutated_gene + ' ' + str(location) + ': ' + mutation_from + ' -> ' + mutation_to)
        if len(mutation_from) == 1 and len(mutation_to) == 1:
            mutation_type = 'snp'
        else:
            if len(mutation_from) > len(mutation_to):
                mutation_type = 'del'
            elif len(mutation_from) < len(mutation_to):
                mutation_type = 'in'
            else:
                mutation_type = 'mnv'

        if mutation_type == 'snp':
            location_on_helix_1 = location_on_gene = location - self.target_genes_start_end_positions[mutated_gene][
                'start']

            if location_on_helix_1 >= 0:  # mutations on genes
                if self.genes[mutated_gene]['gene_complement']:
                    mutation_to = Seq(mutation_to).complement()
                    gene_length = self.target_genes_start_end_positions[mutated_gene]['end'] - \
                                  self.target_genes_start_end_positions[mutated_gene]['start']
                    location_on_gene = gene_length - location_on_helix_1
                else:
                    location_on_gene = location_on_helix_1

                codon_number = int(location_on_gene / 3)
                codon_location = location_on_gene % 3
                codon_from = self.genes[mutated_gene]['codons'][codon_number]
                codon_to = codon_from
                codon_to = codon_to[:codon_location] + mutation_to + codon_to[codon_location + 1:]

                mutation_name = mutated_gene + '_' + self.codon_to_aminoacid[codon_from] + str(codon_number + 1) + \
                                self.codon_to_aminoacid[codon_to]

            else:  # mutations on gene promoters
                if self.genes[mutated_gene]['gene_complement']:
                    mutation_from = Seq(mutation_from).reverse_complement()._data
                    mutation_to = Seq(mutation_to).reverse_complement()._data
                mutation_name = mutated_gene + '_' + mutation_from + str(location_on_gene) + mutation_to
        elif mutation_type == 'in':
            location_on_helix_1 = location_on_gene = location - self.target_genes_start_end_positions[mutated_gene][
                'start']

            if location_on_helix_1 >= 0:  # mutations on genes
                if self.genes[mutated_gene]['gene_complement']:
                    gene_length = self.target_genes_start_end_positions[mutated_gene]['end'] - \
                                  self.target_genes_start_end_positions[mutated_gene]['start']
                    location_on_gene = gene_length - location_on_helix_1
                else:
                    location_on_gene = location_on_helix_1

            if self.genes[mutated_gene]['gene_complement']:
                mutation_from = Seq(mutation_from).reverse_complement()._data
                mutation_to = Seq(mutation_to).reverse_complement()._data
            mutation_name = mutated_gene + '_' + str(location_on_gene) + '_in' + mutation_to[1:]
        elif mutation_type == 'del':
            location_on_helix_1 = location_on_gene = location - self.target_genes_start_end_positions[mutated_gene][
                'start']

            if self.genes[mutated_gene]['gene_complement']:
                gene_length = self.target_genes_start_end_positions[mutated_gene]['end'] - \
                              self.target_genes_start_end_positions[mutated_gene]['start']
                location_on_gene = gene_length - location_on_helix_1
            else:
                location_on_gene = location_on_helix_1

            if self.genes[mutated_gene]['gene_complement']:
                mutation_from = Seq(mutation_from).reverse_complement()._data
                mutation_to = Seq(mutation_to).reverse_complement()._data
            mutation_name = mutated_gene + '_' + str(location_on_gene) + '_ins' + mutation_from[1:]

        else:
            # print(mutation_from + ' -> ' + mutation_to)
            pass

        return mutation_name


def draw_plots(results, model, count=10):
    fig = plt.figure()

    counter = 1
    for drug in results:
        ax = fig.add_subplot(2, 2, counter)
        names = []
        scores = []
        tmp_counter = 0
        for x in results[drug]:
            if tmp_counter < count:
                if len(x['mutation']) < 25:
                    names.append(x['mutation'])
                    scores.append(x['score'])
                    tmp_counter = tmp_counter + 1
        ax.barh(names, scores)
        ax.set_xlabel('SHAP values')
        ax.set_title('SHAP scores for ' + drug)
        ax.invert_yaxis()
        counter = counter + 1

    plt.tight_layout()
    plt.savefig('/home/herkut/Desktop/variants_shap_' + model + '.png')
    plt.show()


if __name__ == '__main__':
    # models = ['rf', 'xgboost']
    models = ['xgboost', 'lr']
    feature_count = 15
    # ar_detector_directory = '/home/herkut/Dekstop/ar_detector'
    ar_detector_directory = '/run/media/herkut/hdd-1/TB_genomes/ar_detector'
    raw = open(os.path.join(ar_detector_directory,
                            'configurations/conf.yml'))
    Config.initialize_configurations(raw)

    # main_directory = '/home/herkut/Desktop/truba/ar_detector_results_dataset-ii_20191205'
    main_directory = '/run/media/herkut/hdd-1/TB_genomes/truba/ar_detector_results_dataset-ii_20191205'

    pp = PostProcessor()

    label_file, raw_feature_selections = get_labels_and_raw_feature_selections('dataset-ii')

    feature_selections = {}
    for k, v in raw_feature_selections.items():
        feature_selections['binary' + '_' + k] = v

    for k, v in feature_selections.items():
        print("Feature importance would be extacted for: " + k)
        raw_label_matrix = FeatureLabelPreparer.get_labels_from_file(os.path.join(Config.dataset_directory, label_file))
        raw_feature_matrix = FeatureLabelPreparer.get_feature_matrix_from_files(v)

    results = {}

    xgboost_results = {}
    lr_results = {}

    for drug in Config.target_drugs:
        for model in models:
            m = None
            most_important_features = None
            most_importance_features_names = None
            most_important_features_scores = None
            if model == 'rf':
                m = RandomForestFeatureExtractor(os.path.join(main_directory,
                                                               'best_models',
                                                               model + '_accuracy_phenotype_binary_snp_09_bcf_nu_indel_00_platypus_all',
                                                               model + '_' + drug + '.sav'),
                                                 raw_feature_matrix.columns)
                most_important_features = m.find_most_important_n_features(feature_count)
            elif model == 'xgboost':
                m = XGBoostFeatureExtractor(os.path.join(main_directory,
                                                         'best_models',
                                                         model + '_accuracy_phenotype_binary_snp_09_bcf_nu_indel_00_platypus_all',
                                                         model + '_' + drug + '.sav'),
                                            raw_feature_matrix.columns,
                                            raw_feature_matrix)
                most_importance_features_names, most_important_features_scores = m.find_most_important_n_features(feature_count)
                most_important_features = []
                for i in range(len(most_importance_features_names)):
                    most_important_features.append([most_importance_features_names[i],
                                                    most_important_features_scores[i]])
            elif model == 'lr':
                m = LogisticRegressionFeatureExtractor(os.path.join(main_directory,
                                                                    'best_models',
                                                                    model + '_accuracy_phenotype_binary_snp_09_bcf_nu_indel_00_platypus_all',
                                                                    model + '_' + drug + '.sav'),
                                                       raw_feature_matrix.columns,
                                                       raw_feature_matrix)
                _, _, most_importance_features_names, most_important_features_scores = m.find_most_important_n_features(feature_count)
                most_important_features = []
                for i in range(len(most_importance_features_names)):
                    most_important_features.append([most_importance_features_names[i],
                                                    most_important_features_scores[i]])
            # print('Found feature importance for: ' + drug)
            important_mutations = pp.find_important_mutations(most_important_features)
            if model == 'xgboost':
                xgboost_results[drug] = important_mutations
            elif model == 'lr':
                lr_results[drug] = important_mutations
                
            save_most_important_features(important_mutations, os.path.join(main_directory,
                                                                           'most_important_features',
                                                                           model + '_' + drug + '.json'))
    if not xgboost_results:
        draw_plots(xgboost_results)
    if not lr_results:
        draw_plots(lr_results)
