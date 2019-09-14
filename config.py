import yaml


class Config:
    dataset_directory = None
    base_directory = None
    results_directory = None

    @classmethod
    def initialize_configurations(cls, cfg):
        raw = yaml.safe_load(cfg)
        cls.dataset_directory = raw['dataset_directory']
        cls.base_directory = raw['base_directory']
        cls.results_directory = raw['results_directory']
