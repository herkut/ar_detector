import collections
import json
import os

from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from utils.confusion_matrix_drawer import plot_confusion_matrix
from utils.numpy_encoder import NumpyEncoder


class ARDetectorByLogisticRegression:
    def __init__(self, results_directory, feature_selection, antibiotic_name, label_tags='phenotype', scoring='roc_auc'):
        self._x_tr = None
        self._y_tr = None
        self._x_te = None
        self._y_te = None
        self._results_directory = results_directory
        self._feature_selection = feature_selection
        self._antibiotic_name = antibiotic_name
        self._label_tags = label_tags
        self._scoring = scoring

        self._model = LogisticRegression()

        self._best_model = None
        self._target_directory = 'lr_' + self._scoring + '_' + self._label_tags + '_' + self._feature_selection

    def initialize_train_dataset(self, x_tr, y_tr):
        self._x_tr = x_tr
        self._y_tr = y_tr

    def initialize_test_dataset(self, x_te, y_te):
        self._x_te = x_te
        self._y_te = y_te

    def initialize_datasets(self, x_tr, y_tr, x_te, y_te):
        self._x_tr = x_tr
        self._y_tr = y_tr
        self._x_te = x_te
        self._y_te = y_te

    def load_model(self):
        # load the model from disk
        self._best_model = joblib.load(self._results_directory + 'best_models/' + self._target_directory + '/lr_' + self._antibiotic_name + '.sav')

    def tune_hyperparameters(self, param_grid):
        model = self._model

        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=self._scoring, cv=5, verbose=True, n_jobs=-1)
        grid.fit(self._x_tr, self._y_tr)

        print(grid)

        if not os.path.exists(self._results_directory + 'grid_search_scores/' + self._target_directory):
            os.makedirs(self._results_directory + 'grid_search_scores/' + self._target_directory)

        with open(self._results_directory + 'grid_search_scores/' + self._target_directory + '/lr_' + self._antibiotic_name + '.json','w') as f:
            f.write(json.dumps(grid.cv_results_, cls=NumpyEncoder))

        # summarize the results of the grid search
        print('Summary of the model:')
        print(grid.best_score_)
        print(grid.best_estimator_.nu)
        print(grid.best_estimator_.gamma)

        self._best_model = grid.best_estimator_

        if not os.path.exists(self._results_directory + 'best_models/' + self._target_directory):
            os.makedirs(self._results_directory + 'best_models/' + self._target_directory)

        # save the model to disk
        filename = self._results_directory + 'best_models/' + self._target_directory + '/lr_' + self._antibiotic_name + '.sav'
        joblib.dump(self._best_model, filename)

    def predict_ar(self, x):
        self._best_model.predict(x)

    def test_model(self):
        y_pred = self._best_model.predict(self._x_te)

        cm = confusion_matrix(self._y_te, y_pred)
        if np.shape(cm)[0] == 2 and np.shape(cm)[1] == 2:
            sensitivity = float(cm[0][0]) / np.sum(cm[0])
            specificity = float(cm[1][1]) / np.sum(cm[1])
            print('For ' + self._antibiotic_name)
            print(collections.Counter(self._y_te))
            print('Sensitivity: ' + str(sensitivity))
            print('Specificity: ' + str(specificity))
        else:
            print('For ' + self._antibiotic_name)
            print('There has been an error in calculating sensitivity and specificity')

        plot_confusion_matrix(self._y_te, y_pred, classes=['susceptible', 'resistant'], normalize=True, title='Normalized confusion matrix')

        if not os.path.exists(self._results_directory + 'confusion_matrices/' + self._target_directory):
            os.makedirs(self._results_directory + 'confusion_matrices/' + self._target_directory)

        plt.savefig(self._results_directory + 'confusion_matrices/' + self._target_directory + '/normalized_lr_' + self._antibiotic_name + '.png')

        plot_confusion_matrix(self._y_te, y_pred, classes=['susceptible', 'resistant'], normalize=False, title='Confusion matrix')

        plt.savefig(self._results_directory + 'confusion_matrices/' + self._target_directory + '/lr_' + self._antibiotic_name + '.png')
