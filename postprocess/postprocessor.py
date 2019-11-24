import os

from config import Config
from postprocess.rf_feature_extractor import RandomForestFeatureExtractor
from preprocess.feature_label_preparer import FeatureLabelPreparer
from run import get_labels_and_raw_feature_selections
from Bio.Seq import Seq


class PostProcessor:
    target_genes = ['ahpC', 'eis', 'embA', 'embB', 'embC', 'embR', 'fabG1', 'gidB', 'gyrA', 'gyrB', 'inhA', 'iniA',
                    'iniC', 'katG', 'manB', 'ndh', 'pncA', 'rmlD', 'rpoB', 'rpsA', 'rpsL', 'rrs', 'tlyA']

    codon_to_aminoacid = {'ATG': 'START',
                          'TAA': 'END', 'TGA': 'END', 'TAG': 'END',
                          'GCT': 'A', 'GCC': 'A', 'GCC': 'A', 'GCG': 'A',
                          'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTT': 'L', 'CTA': 'L', 'CTG': 'L',
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
    gene_codons = None

    def create_codons_for_all_genes(self):
        PostProcessor.gene_codons = {}
        for gene in PostProcessor.target_genes:
            PostProcessor.gene_codons[gene] = self.create_codons_for_gene(gene)

    def create_codons_for_gene(self, gene, additional_base_pair_upstream=100):
        codons = []
        gene_3_to_5 = False
        with open(os.path.join(Config.target_genes_directory, gene)) as f:
            first_line = f.readline()
            elements = first_line.split(' ')
            tmp = elements[0].replace('>', '').split(':')[1]
            tmp_arr = tmp.split('-')
            if tmp_arr[0].startswith('c'):
                gene_3_to_5 = True

            gene_str = ''
            while True:
                line = f.readline().strip()
                gene_str += line
                if not line:
                    break

            if gene_3_to_5:
                counter = 0
                tmp_codon = ''
                """
                for nucleotide in reversed(gene_str):
                    if counter > 2:
                        counter = 0
                        codons.append(tmp_codon)
                        tmp_codon = nucleotide
                    else:
                        tmp_codon += nucleotide
                    counter += 1
                """
                za = Seq(gene_str)
                for nucleotide in za:
                    if counter > 2:
                        counter = 0
                        codons.append(tmp_codon)
                        tmp_codon = nucleotide
                    else:
                        tmp_codon += nucleotide
                    counter += 1

            else:
                counter = 0
                tmp_codon = ''
                for nucleotide in gene_str:
                    if counter > 2:
                        counter = 0
                        codons.append(tmp_codon)
                        tmp_codon = nucleotide
                    else:
                        tmp_codon += nucleotide
                    counter += 1
        return codons

    def load_start_and_end_positions_for_all_target_genes(self, additional_base_pair_upstream=100):
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

    def find_mutated_gene_from_mutation_key(self, mutation_key):
        mutation_location = self.get_mutation_location_from_mutation_key(mutation_key)

        mutated_gene = None
        for gene in PostProcessor.target_genes_start_end_positions:
            if PostProcessor.target_genes_start_end_positions[gene]['start'] < mutation_location < PostProcessor.target_genes_start_end_positions[gene]['end']:
                mutated_gene = gene
                break

        return mutated_gene

    def __init__(self):
        self.load_start_and_end_positions_for_all_target_genes()
        self.create_codons_for_all_genes()


if __name__ == '__main__':
    raw = open('/run/media/herkut/hdd-1/TB_genomes/ar_detector/configurations/conf.yml')
    Config.initialize_configurations(raw)

    pp = PostProcessor()

    print(pp.gene_codons['katG'][313] + ' - ' + pp.codon_to_aminoacid[pp.gene_codons['katG'][313]])
    print(pp.gene_codons['katG'][314] + ' - ' + pp.codon_to_aminoacid[pp.gene_codons['katG'][314]])
    print(pp.gene_codons['katG'][315] + ' - ' + pp.codon_to_aminoacid[pp.gene_codons['katG'][315]])
    print(pp.gene_codons['katG'][316] + ' - ' + pp.codon_to_aminoacid[pp.gene_codons['katG'][316]])

    print(pp.gene_codons['fabG1'][13] + ' - ' + pp.codon_to_aminoacid[pp.gene_codons['fabG1'][13]])
    print(pp.gene_codons['fabG1'][14] + ' - ' + pp.codon_to_aminoacid[pp.gene_codons['fabG1'][14]])
    print(pp.gene_codons['fabG1'][15] + ' - ' + pp.codon_to_aminoacid[pp.gene_codons['fabG1'][15]])
    print(pp.gene_codons['fabG1'][16] + ' - ' + pp.codon_to_aminoacid[pp.gene_codons['fabG1'][16]])

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

        most_importance_features = rf.find_most_important_n_features(50)
        print('Found feature importance for: ' + drug)
        for mif in most_importance_features:
            print(pp.find_mutated_gene_from_mutation_key(mif[0]) + ' at ' + str(mif[0].split('_')[0]) +
                  ' mutation ' + str(mif[0].split('_')[1]) + ' -> ' + str(mif[0].split('_')[2]) +
                  ' with score ' + str(mif[1]))

    print(pp.find_mutated_gene_from_mutation_key('1917972_A_G'))

    print('Zaa')
