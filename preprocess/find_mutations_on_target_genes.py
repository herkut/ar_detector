import os
import re
import yaml
import subprocess
import logging
import pandas as pd
import numpy as np
from preprocess.variants_finder import VariantsFinder

#############################################################################33
TARGET_GENES_DIRECTORY = '/run/media/herkut/herkut/TB_genomes/target_genes/'
GENOMES_DIRECTORY = '/run/media/herkut/herkut/TB_genomes/genomes/'
BCFTOOLS = '/home/herkut/Desktop/TB_genomes/tools/bcftools-1.9/bin/bcftools'
BGZIP = '/home/herkut/Desktop/TB_genomes/tools/htslib-1.9/bin/bgzip'
TABIX = '/home/herkut/Desktop/TB_genomes/tools/htslib-1.9/bin/tabix'
CHROMOSOME = 'NC_000962.3'
MUTATIONS_TARGET_DIRECTORY = '/run/media/herkut/herkut/TB_genomes/ar_detection_dataset/new_approach/'
MUTATIONS_FILE_PREFIX = 'feature_matrix_09_'
MUTATIONS_FILE_LIKE_BASELINE_PREFIX = 'feature_matrix_like_baseline_'
#############################################################################33


class FindMutationsOnTargetGenes:
    MUTATIONS = None
    MUTATIONS_LIKE_BASELINE = None
    variant_finder = None

    @staticmethod
    def initialize(target_directories):
        index = list(map(lambda x: int(x), target_directories))
        FindMutationsOnTargetGenes.MUTATIONS = pd.DataFrame(index=index)

        index = list(map(lambda x: int(x), target_directories))
        FindMutationsOnTargetGenes.MUTATIONS_LIKE_BASELINE = pd.DataFrame(index=index)

        thresholds = [0.9, 0.75, 0.0]
        FindMutationsOnTargetGenes.variant_finder = VariantsFinder(target_directories, thresholds)

    """
        Find variant type, it may be one of the followings: SNP, INDEL
        If length of sequence in reference genome and the variant is 1 
        then it is assumed as SNP(single nucleotid polymorphism);
        otherwise it is assumed as INDEL
    """
    @staticmethod
    def get_variant_type(seq_in_ref, seq_in_variant):
        if len(seq_in_ref) == 1 and len(seq_in_variant) == 1:
            return 'SNP'
        else:
            return 'INDEL'

    @staticmethod
    def collect_all_mutations_on_target_genes():
        print('')

    @staticmethod
    def find_bcftools_queries_for_genes(additional_base_pair_upstream=0):
        genes_list = []
        genes_list += [each for each in os.listdir(TARGET_GENES_DIRECTORY)]

        genes_queries = {}

        for gene in genes_list:
            with open(TARGET_GENES_DIRECTORY + gene) as f:
                first_line = f.readline()
                elements = first_line.split(' ')
                # replacing 'c' with '' some target genes containing c character just before their positions on the DNA
                tmp = elements[0].replace('>', '').replace('c', '')
                tmp_arr = tmp.split(':')[1].split('-')
                if (int(tmp_arr[0]) - additional_base_pair_upstream) < 0:
                    tmp_arr[0] = str(0)
                else:
                    tmp_arr[0] = str(int(tmp_arr[0]) - additional_base_pair_upstream)

                genes_queries[gene] = tmp.split(':')[0] + ':' + tmp_arr[0] + '-' + tmp_arr[1]

        return genes_queries

    @staticmethod
    def bgzip_vcf(target_directory, vcf_file):
        logger = logging.getLogger("genome_assembler_logger")

        is_successfully_completed = True

        command = BGZIP + ' -fc ' + target_directory + vcf_file + ' > ' + target_directory + vcf_file + '.gz'
        try:
            print(command)
            proc = subprocess.Popen(command, shell=True)
            output, unused_err = proc.communicate()
            retcode = proc.poll()
            # subprocess.check_output(command, shell=True)
        except subprocess.CalledProcessError as e:
            logger.error('An error occurred while bgzipping the vcf: ' + target_directory + vcf_file)
            logger.error(e.output)
            is_successfully_completed = False
        return is_successfully_completed

    @staticmethod
    def index_vcf(target_directory, vcf_file):
        logger = logging.getLogger("genome_assembler_logger")

        is_successfully_completed = True

        command = TABIX + ' -fp vcf ' + target_directory + vcf_file + '.gz'
        try:
            print(command)
            proc = subprocess.Popen(command, shell=True)
            output, unused_err = proc.communicate()
            retcode = proc.poll()
            # subprocess.check_output(command, shell=True)
        except subprocess.CalledProcessError as e:
            logger.error('An error occurred while indexing the vcf: ' + target_directory + vcf_file)
            logger.error(e.output)
            is_successfully_completed = False
        return is_successfully_completed

    @staticmethod
    def find_annotated_vcf_files(target_directory):
        logger = logging.getLogger("genome_assembler_logger")
        vcfs_to_index = []
        files_to_find_mutation_in = []
        # gzip and index vcfs to properly find mutations
        vcfs_to_index += [each for each in os.listdir(target_directory) if each.endswith('.vcf') and each.startswith('annotated_calls_via')]
        for vcf_file in vcfs_to_index:
            print('bgzipping')
            if not FindMutationsOnTargetGenes.bgzip_vcf(target_directory, vcf_file):
                logger.error('An error occurred while bgzipping the vcf: ' + target_directory + vcf_file + ', manual operation may be required')
                break
            print('indexing')
            if not FindMutationsOnTargetGenes.index_vcf(target_directory, vcf_file):
                logger.error('An error occurred while indexing the vcf: ' + target_directory + vcf_file + ', manual operation may be required')
                break

        files_to_find_mutation_in += [each for each in os.listdir(target_directory) if each.endswith('.vcf.gz') and each.startswith('annotated_calls_via')]
        return files_to_find_mutation_in

    @staticmethod
    def find_isolate_id_from_directory_name(target_directory):
        elements = target_directory.split('/')
        return elements[-2]

    @staticmethod
    def extract_mutation_key(str_containing_mutation, tool='bcftools', threshold=0.9):
        mutation_keys = []
        elements = re.split(r'\t', str_containing_mutation)

        if tool == 'bcftools':
            key_values = elements[7].split(';')

            divider = 1
            dividend = 1

            for key_value in key_values:
                tmp_arr = key_value.split('=')
                if tmp_arr[0] == 'DP':
                    divider = int(tmp_arr[1])
                elif tmp_arr[0] == 'DP4':
                    dividend = int(tmp_arr[1].split(',')[2])
                    dividend += int(tmp_arr[1].split(',')[3])

            if (float(dividend) / divider) >= threshold:
                mutation_keys.append(elements[1] + '_' + elements[3] + '_' + elements[4])

        elif tool == 'platypus':
            potential_mutations = elements[4].split(',')

            key_values = elements[7].split(';')

            dividend = np.ones(len(potential_mutations))

            divider = 1

            for key_value in key_values:
                # In some isolates ALT is like CCCCAAGGG,AAATTTTT in this type of isolates TR also has values seperated by comma
                for i in range(len(potential_mutations)):
                    tmp_arr = key_value.split('=')
                    if tmp_arr[0] == 'TR':
                        dividend[i] = int(tmp_arr[1].split(',')[i])
                    elif tmp_arr[0] == 'TC':
                        divider = int(tmp_arr[1])
            for i in range(len(potential_mutations)):
                if float(dividend[i]) / divider >= threshold:
                    mutation_keys.append(elements[1] + '_' + elements[3] + '_' + potential_mutations[i])

        return mutation_keys

    @staticmethod
    def extract_mutation_key_like_in_baseline(str_containing_mutation, tool='bcftools', threshold=0.9):
        mutation_keys = []
        elements = re.split(r'\t', str_containing_mutation)

        # Ignore Indels, apply read depth filter
        if tool == 'bcftools':
            variant_type = FindMutationsOnTargetGenes.get_variant_type(elements[3], elements[4])

            if variant_type == 'SNP':
                key_values = elements[7].split(';')

                divider = 1
                dividend = 1

                for key_value in key_values:
                    tmp_arr = key_value.split('=')
                    if tmp_arr[0] == 'DP':
                        divider = int(tmp_arr[1])
                    elif tmp_arr[0] == 'DP4':
                        dividend = int(tmp_arr[1].split(',')[2])
                        dividend += int(tmp_arr[1].split(',')[3])

                if (float(dividend) / divider) >= threshold:
                    mutation_keys.append(elements[1] + '_' + elements[3] + '_' + elements[4])

        # Ignore SNPs and not apply read depth filter
        elif tool == 'platypus':
            variant_types = []
            potential_mutations = elements[4].split(',')

            for potential_mutation in potential_mutations:
                variant_types.append(potential_mutation)

            for i in range(len(potential_mutations)):
                if variant_types[i] == 'INDEL':
                    mutation_keys.append(elements[1] + '_' + elements[3] + '_' + potential_mutations[i])

        return mutation_keys

    @staticmethod
    def find_mutations_on_target_genes(target_directory):
        logger = logging.getLogger("genome_assembler_logger")

        genes_queries = FindMutationsOnTargetGenes.find_bcftools_queries_for_genes(additional_base_pair_upstream=100)

        files_to_find_mutation_in = FindMutationsOnTargetGenes.find_annotated_vcf_files(target_directory)

        isolate_id = int(FindMutationsOnTargetGenes.find_isolate_id_from_directory_name(target_directory))

        #if not isolate_id in FindMutationsOnTargetGenes.MUTATIONS.index:
        #    FindMutationsOnTargetGenes.MUTATIONS.loc[isolate_id] = [0 for n in len(FindMutationsOnTargetGenes.MUTATIONS.columns)]

        for gene, query in genes_queries.iteritems():
            print(gene + ' with query: ' + query)
            for file in files_to_find_mutation_in:
                command = BCFTOOLS + ' view ' + target_directory + file + ' -t ' + query + ' | grep \"^' + CHROMOSOME + '\"'
                try:
                    #print(command)
                    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
                    output, unused_err = proc.communicate()
                    for line in output.split(os.linesep):
                        if line != '' and line != '[W::bcf_hdr_check_sanity] GL should be declared as Number=G':
                            FindMutationsOnTargetGenes.variant_finder.find_variants(isolate_id, line)

                    retcode = proc.poll()
                    # subprocess.check_output(command, shell=True)
                except subprocess.CalledProcessError as e:
                    logger.error('An error occured while finding mutations on gene:  ' + gene + ' with query ' + query + ' for ' + target_directory + file)
                    logger.error(e.output)
                    is_successfully_completed = False

    @staticmethod
    def filter_mutations_occurred_only_once():
        removed_mutations = []
        # TODO check whether = asigns address or it copies the original panda frame
        mutations_without_unique_ones = FindMutationsOnTargetGenes.MUTATIONS.copy(deep=True)
        for column in FindMutationsOnTargetGenes.MUTATIONS:
            x = mutations_without_unique_ones[column].value_counts()
            if x[1] <= 1:
                removed_mutations.append(column)
                mutations_without_unique_ones.drop(column, 1, inplace=True)
        return mutations_without_unique_ones, removed_mutations

    @staticmethod
    def filter_mutations_occurred_only_once_like_baseline():
        removed_mutations = []
        # TODO check whether = asigns address or it copies the original panda frame
        mutations_without_unique_ones = FindMutationsOnTargetGenes.MUTATIONS_LIKE_BASELINE.copy(deep=True)
        for column in FindMutationsOnTargetGenes.MUTATIONS_LIKE_BASELINE:
            x = mutations_without_unique_ones[column].value_counts()
            if x[1] <= 1 and FindMutationsOnTargetGenes.get_variant_type(column.split('_')[1], column.split('_')[2]) == 'SNP':
                removed_mutations.append(column)
                mutations_without_unique_ones.drop(column, 1, inplace=True)
        return mutations_without_unique_ones, removed_mutations

    @staticmethod
    def save_all_mutations():
        FindMutationsOnTargetGenes.variant_finder.save_all_variants_into_file(MUTATIONS_TARGET_DIRECTORY)

    @staticmethod
    def save_all_mutations_including_baseline_approach(all_mutations, mutations_without_unique_ones, all_mutations_like_baseline, mutations_without_unique_ones_like_baseline):
        all_mutations.to_csv(r'' + MUTATIONS_TARGET_DIRECTORY + MUTATIONS_FILE_PREFIX + 'with_all_mutations.csv', index=True, header=True)
        mutations_without_unique_ones.to_csv(r'' + MUTATIONS_TARGET_DIRECTORY + MUTATIONS_FILE_PREFIX + 'without_unique_mutations.csv', index=True,header=True)

        all_mutations_like_baseline.to_csv(r'' + MUTATIONS_TARGET_DIRECTORY + MUTATIONS_FILE_LIKE_BASELINE_PREFIX + 'with_all_mutations.csv', index=True,header=True)
        mutations_without_unique_ones_like_baseline.to_csv(r'' + MUTATIONS_TARGET_DIRECTORY + MUTATIONS_FILE_LIKE_BASELINE_PREFIX + 'without_unique_mutations.csv', index=True, header=True)
