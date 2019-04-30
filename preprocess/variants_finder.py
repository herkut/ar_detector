import re
import numpy as np
import pandas as pd

MUTATIONS_TARGET_DIRECTORY = '/run/media/herkut/herkut/TB_genomes/ar_detection_dataset'


class VariantsFinder:

    def __init__(self, target_directories, thresholds):
        self.thresholds = thresholds
        self.thresholds_variants = {}
        for threshold in thresholds:
            tmp_dict = {}
            index = list(map(lambda x: int(x), target_directories))
            tmp_dict['snp_bcftools'] = pd.DataFrame(index=index)

            index = list(map(lambda x: int(x), target_directories))
            tmp_dict['snp_platypus'] = pd.DataFrame(index=index)

            index = list(map(lambda x: int(x), target_directories))
            tmp_dict['indel_bcftools'] = pd.DataFrame(index=index)

            index = list(map(lambda x: int(x), target_directories))
            tmp_dict['indel_platypus'] = pd.DataFrame(index=index)

            self.thresholds_variants[str(threshold)] = tmp_dict

    def get_variant_type(self, seq_in_ref, seq_in_variant):
        if len(seq_in_ref) == 1 and len(seq_in_variant) == 1:
            return 'SNP'
        else:
            return 'INDEL'

    def find_variants(self, isolate_id, variant_line):
        for threshold in self.thresholds_variants:
            # Find SNPs
            self.find_snps(isolate_id, variant_line, 'bcftools', float(threshold))
            self.find_snps(isolate_id, variant_line, 'platypus', float(threshold))

            # Find INDELs
            self.find_indels(isolate_id, variant_line, 'bcftools', float(threshold))
            self.find_indels(isolate_id, variant_line, 'platypus', float(threshold))

    def find_snps(self, isolate_id, variant_line, tool='bcftools', threshold=0.9):
        mutation_keys = []
        elements = re.split(r'\t', variant_line)

        if tool == 'bcftools':
            key_values = elements[7].split(';')
            variant_type = self.get_variant_type(elements[3], elements[4])

            if variant_type == 'SNP':
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
                    mutation_key = elements[1] + '_' + elements[3] + '_' + elements[4]
                    if mutation_key not in self.thresholds_variants[str(threshold)]['snp_bcftools'].columns:
                        self.thresholds_variants[str(threshold)]['snp_bcftools'][mutation_key] = 0

                    self.thresholds_variants[str(threshold)]['snp_bcftools'].at[isolate_id, mutation_key] = 1

        elif tool == 'platypus':
            potential_mutations = elements[4].split(',')
            variant_types = []

            for potential_mutation in potential_mutations:
                variant_types.append(self.get_variant_type(elements[3], potential_mutation))

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
                    if variant_types[i] == 'SNP':
                        mutation_key = elements[1] + '_' + elements[3] + '_' + potential_mutations[i]
                        if mutation_key not in self.thresholds_variants[str(threshold)]['snp_platypus'].columns:
                            self.thresholds_variants[str(threshold)]['snp_platypus'][mutation_key] = 0

                        self.thresholds_variants[str(threshold)]['snp_platypus'].at[isolate_id, mutation_key] = 1

    def find_indels(self, isolate_id, variant_line, tool='bcftools', threshold=0.9):
        mutation_keys = []
        elements = re.split(r'\t', variant_line)

        if tool == 'bcftools':
            key_values = elements[7].split(';')
            variant_type = self.get_variant_type(elements[3], elements[4])

            if variant_type == 'INDEL':
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
                    mutation_key = elements[1] + '_' + elements[3] + '_' + elements[4]
                    if mutation_key not in self.thresholds_variants[str(threshold)]['indel_bcftools'].columns:
                        self.thresholds_variants[str(threshold)]['indel_bcftools'][mutation_key] = 0

                    self.thresholds_variants[str(threshold)]['indel_bcftools'].at[isolate_id, mutation_key] = 1

        elif tool == 'platypus':
            potential_mutations = elements[4].split(',')
            variant_types = []

            for potential_mutation in potential_mutations:
                variant_types.append(self.get_variant_type(elements[3], potential_mutation))

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
                    if variant_types[i] == 'INDEL':
                        mutation_key = elements[1] + '_' + elements[3] + '_' + potential_mutations[i]
                        if mutation_key not in self.thresholds_variants[str(threshold)]['indel_platypus'].columns:
                            self.thresholds_variants[str(threshold)]['indel_platypus'][mutation_key] = 0

                        self.thresholds_variants[str(threshold)]['indel_platypus'].at[isolate_id, mutation_key] = 1

    def filter_mutations_occurred_only_once(self, variants):
        removed_mutations = []
        # TODO check whether = asigns address or it copies the original panda frame
        mutations_without_unique_ones = variants.copy(deep=True)
        for column in variants:
            x = mutations_without_unique_ones[column].value_counts()
            if x[1] <= 1:
                removed_mutations.append(column)
                mutations_without_unique_ones.drop(column, 1, inplace=True)
        return mutations_without_unique_ones, removed_mutations

    def save_all_variants_into_file(self, parent_target_directory):
        for threshold, variants in self.thresholds_variants.items():
            variants['snp_bcftools'].to_csv(r'' + parent_target_directory + 'snp_bcftools_' + str(threshold) + '_all.csv', index=True, header=True)
            x, _ = self.filter_mutations_occurred_only_once(variants['snp_bcftools'])
            x.to_csv(r'' + parent_target_directory + 'snp_bcftools_' + str(threshold) + '_notunique.csv', index=True, header=True)

            variants['snp_platypus'].to_csv(r'' + parent_target_directory + 'snp_platypus_' + str(threshold) + '_all.csv', index=True, header=True)
            x, _ = self.filter_mutations_occurred_only_once(variants['snp_platypus'])
            x.to_csv(r'' + parent_target_directory + 'snp_platypus_' + str(threshold) + '_notunique.csv', index=True, header=True)

            variants['indel_bcftools'].to_csv(r'' + parent_target_directory + 'indel_bcftools_' + str(threshold) + '_all.csv', index=True, header=True)
            x, _ = self.filter_mutations_occurred_only_once(variants['indel_bcftools'])
            x.to_csv(r'' + parent_target_directory + 'indel_bcftools_' + str(threshold) + '_notunique.csv', index=True, header=True)

            variants['indel_platypus'].to_csv(r'' + parent_target_directory + 'indel_platypus_' + str(threshold) + '_all.csv', index=True, header=True)
            x, _ = self.filter_mutations_occurred_only_once(variants['indel_platypus'])
            x.to_csv(r'' + parent_target_directory + 'indel_platypus_' + str(threshold) + '_notqunique.csv', index=True, header=True)
