import yaml


class Config:
    dataset_directory = None
    base_directory = None
    results_directory = None
    dataset_index_directory = None
    traditional_ml_scoring = None
    label_tags = None
    target_drugs = None
    scikit_learn_n_jobs = None

    @classmethod
    def initialize_configurations(cls, cfg):
        raw = yaml.safe_load(cfg)
        cls.dataset_directory = raw['dataset_directory']
        cls.base_directory = raw['base_directory']
        cls.results_directory = raw['results_directory']
        cls.dataset_index_directory = raw['dataset_index_directory']
        cls.traditional_ml_scoring = raw['traditional_ml_scoring']
        cls.label_tags = raw['label_tags']
        cls.scikit_learn_n_jobs = raw['n_jobs']

        cls.target_drugs = ['Isoniazid', 'Rifampicin', 'Ethambutol', 'Pyrazinamide', 'Streptomycin', 'Ofloxacin',
                                'Amikacin', 'Ciprofloxacin', 'Moxifloxacin', 'Capreomycin', 'Kanamycin']