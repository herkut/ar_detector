import yaml


class Config:
    dataset_directory = None
    cnn_dataset_directory = None
    cnn_feature_size = None
    cnn_first_in_channel = None
    cnn_output_size = None
    base_directory = None
    results_directory = None
    hyperparameter_grids_directory = None
    dataset_index_directory = None
    target_dataset = None
    cnn_target_dataset = None
    traditional_ml_scoring = None
    label_tags = None
    target_drugs = None
    scikit_learn_n_jobs = None
    deep_learning_metric = None

    @classmethod
    def initialize_configurations(cls, cfg):
        raw = yaml.safe_load(cfg)
        cls.dataset_directory = raw['dataset_directory']
        cls.cnn_dataset_directory = raw['cnn_dataset_directory']
        cls.cnn_feature_size = raw['cnn_feature_size']
        cls.cnn_first_in_channel = raw['cnn_first_in_channel']
        cls.cnn_output_size = raw['cnn_output_size']
        cls.base_directory = raw['base_directory']
        cls.results_directory = raw['results_directory']
        cls.dataset_index_directory = raw['dataset_index_directory']
        cls.target_dataset = raw['target_dataset']
        cls.cnn_target_dataset = raw['cnn_target_dataset']
        cls.traditional_ml_scoring = raw['traditional_ml_scoring']
        cls.label_tags = raw['label_tags']
        cls.scikit_learn_n_jobs = raw['n_jobs']
        cls.hyperparameter_grids_directory = raw['hyperparameter_grids_directory']
        cls.deep_learning_metric = raw['deep_learning_metric']

        cls.target_drugs = ['Isoniazid', 'Rifampicin', 'Ethambutol', 'Pyrazinamide']

    @classmethod
    def set_target_drugs(cls, target_drugs):
        cls.target_drugs = target_drugs
