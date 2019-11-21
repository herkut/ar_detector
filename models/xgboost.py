import json
import os

import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from config import Config
from models.base_ar_detector import BaseARDetector
from xgboost import XGBClassifier
import pandas as pd

from utils.confusion_matrix_drawer import plot_confusion_matrix
from utils.helper_functions import get_k_fold
from utils.numpy_encoder import NumpyEncoder


class ARDetectorByXGBoost(BaseARDetector):
    def __init__(self, feature_selection, dataset, antibiotic_name=None, class_weights=None):
        self._results_directory = Config.results_directory
        self._feature_selection = feature_selection
        self._dataset = dataset
        self._results_directory = self._results_directory + '_' + self._dataset
        self._label_tags = Config.label_tags
        self._model_name = 'xgboost'
        self._class_weights = class_weights

        self._model = XGBClassifier(objective='binary:logistic', scale_pos_weight=self._class_weights[1])

        self._best_model = None
        self._antibiotic_name = antibiotic_name
        self._scoring = Config.traditional_ml_scoring
        self._target_directory = 'xgboost_' + self._scoring + '_' + self._label_tags + '_' + self._feature_selection

    def set_antibiotic_name(self, antibiotic_name):
        self._antibiotic_name = antibiotic_name

    def reinitialize_model_with_parameters(self, parameters):
        self._model = XGBClassifier(parameters['max_depth'],
                                    parameters['n_estimators'],
                                    parameters['learning_rate'],
                                    objective='binary:logistic',
                                    scale_pos_weight=self._class_weights[1])

    def reinitialize_best_model_with_parameters(self, parameters):
        self._best_model = XGBClassifier(parameters['max_depth'],
                                         parameters['n_estimators'],
                                         parameters['learning_rate'],
                                         objective='binary:logistic',
                                         scale_pos_weight=self._class_weights[1])

    def load_model(self):
        self._best_model = joblib.load(os.path.join(self._results_directory,
                                                    'best_models',
                                                    self._target_directory,
                                                    self._model_name + '_' + self._antibiotic_name + '.sav'))

    def tune_hyperparameters(self, param_grid, x_tr, y_tr):
        model = self._model

        cv = get_k_fold(10)

        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            scoring=self._scoring,
                            cv=cv,
                            verbose=True,
                            n_jobs=Config.scikit_learn_n_jobs)
        grid.fit(x_tr, y_tr)

        print(grid)

        if not os.path.exists(os.path.join(self._results_directory, 'grid_search_scores', self._target_directory)):
            os.makedirs(os.path.join(self._results_directory, 'grid_search_scores', self._target_directory))

        with open(os.path.join(self._results_directory, 'grid_search_scores', self._target_directory, 'xgboost' + self._antibiotic_name + '.json'), 'w') as f:
            f.write(json.dumps(grid.cv_results_, cls=NumpyEncoder))

        # summarize the results of the grid search
        if not os.path.exists(os.path.join(self._results_directory, 'best_models', self._target_directory)):
            os.makedirs(os.path.join(self._results_directory, 'best_models', self._target_directory))

        with open(os.path.join(self._results_directory,
                               'best_models',
                               self._target_directory,
                               self._model_name + '_' + self._antibiotic_name + '.json'), 'w') as f:
            f.write(json.dumps(grid.best_params_, cls=NumpyEncoder))

        print('Summary of the model:')
        print(grid.best_score_)
        print(grid.best_estimator_.max_depth)
        print(grid.best_estimator_.n_estimators)
        print(grid.best_estimator_.learning_rate)

    def predict_ar(self, x):
        self._best_model.predict(x)

    def predict(self, x):
        return self._model.predict(x)

    def train_model(self, x_tr, y_tr):
        self._model.fit(x_tr, y_tr)

    def train_best_model(self, hyperparameters, x_tr, y_tr, x_te, y_te):
        self.reinitialize_best_model_with_parameters(hyperparameters)
        self._best_model.fit(x_tr, y_tr)

        if not os.path.exists(os.path.join(self._results_directory, 'best_models', self._target_directory)):
            os.makedirs(os.path.join(self._results_directory, 'best_models', self._target_directory))

        # save the model to disk
        filename = os.path.join(self._results_directory,
                                'best_models',
                                self._target_directory,
                                self._model_name + '_' + self._antibiotic_name + '.sav')
        joblib.dump(self._best_model, filename)

    def test_model(self, x_te, y_te):
        y_pred = self._best_model.predict(x_te)

        plot_confusion_matrix(y_te, y_pred, classes=['susceptible', 'resistant'], normalize=True, title='Normalized confusion matrix')

        if not os.path.exists(os.path.join(self._results_directory, 'confusion_matrices', self._target_directory)):
            os.makedirs(os.path.join(self._results_directory, 'confusion_matrices', self._target_directory))

        plt.savefig(os.path.join(self._results_directory,
                                 'confusion_matrices',
                                 self._target_directory,
                                 'normalized_' + self._model_name + '_' + self._antibiotic_name + '.png'))

        plot_confusion_matrix(y_te,
                              y_pred,
                              classes=['susceptible', 'resistant'],
                              normalize=False,
                              title='Confusion matrix')

        plt.savefig(os.path.join(self._results_directory,
                                 'confusion_matrices',
                                 self._target_directory,
                                 self._model_name + '_' + self._antibiotic_name + '.png'))

        y_true = pd.Series(y_te, name="Actual")
        y_pred = pd.Series(y_pred, name="Predicted")
        df_confusion = pd.crosstab(y_true, y_pred)
        df_confusion.to_csv(os.path.join(self._results_directory,
                                         'confusion_matrices',
                                         self._target_directory,
                                         self._model_name + '_' + self._antibiotic_name + '.csv'))