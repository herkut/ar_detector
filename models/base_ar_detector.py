class BaseARDetector:

    def load_model(self):
        raise NotImplementedError('Subclasses must override load_model')

    def save_model(self):
        raise NotImplementedError('Subclasses must override load_model')

    def tune_hyperparameters(self, param_grid):
        raise NotImplementedError('Subclasses must override tune_hyperparameters')

    def predict_ar(self, x):
        raise NotImplementedError('Subclasses must override predict_ar')

    def test_model(self):
        raise NotImplementedError('Subclasses must override test_model')
