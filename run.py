from docopt import docopt

from preprocess.find_mutations_on_target_genes import FindMutationsOnTargetGenes
from utils.input_parser import InputParser

"""
 Run the following command to concatenate all files in rerun_pipeline.txt
 cat Desktop/rerun_pipeline.txt | sed ':a;N;$!ba;s/\n/,/g'
"""


def main():
    args = docopt("""
    Usage: 
        run.py find_mutations <target_base_directory> <target_directory_ids>
        run.py prepare_features <configuration_file> <target_base_directory>
        run.py train_models <configuration_file> <target_base_directory>

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

        mutations_without_unique_ones, mutations_observed_only_ones = FindMutationsOnTargetGenes.filter_mutations_occurred_only_once()

        mutations_without_unique_ones_like_baseline, mutations_observed_only_ones_like_baseline = FindMutationsOnTargetGenes.filter_mutations_occurred_only_once_like_baseline()

        FindMutationsOnTargetGenes.save_all_mutations_including_baseline_approach(FindMutationsOnTargetGenes.MUTATIONS, mutations_without_unique_ones, FindMutationsOnTargetGenes.MUTATIONS_LIKE_BASELINE, mutations_without_unique_ones_like_baseline)


if __name__ == '__main__':
    main()
