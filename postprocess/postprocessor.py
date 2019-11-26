import os

from config import Config
from postprocess.rf_feature_extractor import RandomForestFeatureExtractor
from preprocess.feature_label_preparer import FeatureLabelPreparer
from run import get_labels_and_raw_feature_selections
from Bio.Seq import Seq
from Bio import SeqIO
import math
import json


class PostProcessor:
    target_genes = ['ahpC', 'eis', 'embA', 'embB', 'embC', 'embR', 'fabG1', 'gidB', 'gyrA', 'gyrB', 'inhA', 'iniA',
                    'iniC', 'katG', 'manB', 'ndh', 'pncA', 'rmlD', 'rpoB', 'rpsA', 'rpsL', 'rrs', 'tlyA']

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
            if PostProcessor.target_genes_start_end_positions[gene]['start']-additional_base_pair_upstream < mutation_location < PostProcessor.target_genes_start_end_positions[gene]['end']:
                mutated_gene = gene
                break

        return mutated_gene

    def __init__(self):
        self.reference_genome = list(SeqIO.parse('/run/media/herkut/herkut/TB_genomes/reference_genome/mtb_h37rv_v3.fasta', 'fasta'))[0]._seq
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
            if len(mutation_from) == 1 and len(mutation_to) == 1:
                mutation_type = 'snp'
            else:
                mutation_type = 'indel'

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
            else:
                # print('Mutation is an indel not a snp: ' + mutation_from + ' -> ' + mutation_to)
                pass

        return important_mutations


if __name__ == '__main__':
    raw = open('/home/herkut/Desktop/ar_detector/configurations/conf.yml')
    Config.initialize_configurations(raw)

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

    for drug in Config.target_drugs:
        rf = RandomForestFeatureExtractor(os.path.join('/home/herkut/Desktop/truba/ar_detector_results_dataset-ii_20191118',
                                                       'best_models',
                                                       'rf_accuracy_phenotype_binary_snp_09_bcf_nu_indel_00_platypus_all',
                                                       'rf_' + drug + '.sav'),
                                          raw_feature_matrix.columns)

        most_importance_features = rf.find_most_important_n_features(20)
        # print('Found feature importance for: ' + drug)
        important_mutations = pp.find_important_mutations(most_importance_features)

        with open(os.path.join('/home/herkut/Desktop/truba/ar_detector_results_dataset-ii_20191118',
                               'most_important_features',
                               'rf_' + drug + '.json'), 'w') as file:
            file.write(json.dumps(important_mutations))  # use `json.loads` to do the reverse
