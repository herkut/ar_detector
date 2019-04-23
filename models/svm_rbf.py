import collections
import json
import os

import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from utils.confusion_matrix_drawer import plot_confusion_matrix
from utils.numpy_encoder import NumpyEncoder


class ARDetectorBySVMWithRBF:
    def __init__(self, target_base_directory, feature_selection, antibiotic_name, label_tags='phenotype', scoring='roc_auc'):
        self._x_tr = None
        self._y_tr = None
        self._x_te = None
        self._y_te = None
        self._target_base_directory = target_base_directory
        self._feature_selection = feature_selection
        self._label_tags = label_tags
        self._model = svm.SVC(kernel='rbf')
        self._best_model = None
        self._antibiotic_name = antibiotic_name
        self._scoring = scoring

    def initialize_datasets(self, x_tr, y_tr, x_te, y_te):
        self._x_tr = x_tr
        self._y_tr = y_tr
        self._x_te = x_te
        self._y_te = y_te

    def load_model(self):
        #load the model from disk
        target_directory = self._scoring + '_' + self._label_tags + '_' + self._feature_selection
        self._best_model = joblib.load(self._target_base_directory + 'best_models/' + target_directory + '/svm_rbf_model_for_' + self._antibiotic_name + '.sav')

    def tune_hyperparameters(self, c, gamma):
        param_grid = {'C': c, 'gamma': gamma}
        model = self._model

        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=self._scoring, cv=5, verbose=True, n_jobs=-1)
        grid.fit(self._x_tr, self._y_tr)

        print(grid)

        target_directory = self._scoring + '_' + self._label_tags + '_' + self._feature_selection

        if not os.path.exists(self._target_base_directory + 'grid_search_scores/' + target_directory):
            os.makedirs(self._target_base_directory + 'grid_search_scores/' + target_directory)

        with open(self._target_base_directory + 'grid_search_scores/' + target_directory + '/svm_rbf_' + self._antibiotic_name + '.json', 'w') as f:
            f.write(json.dumps(grid.cv_results_, cls=NumpyEncoder))

        # summarize the results of the grid search
        print('Summary of the model:')
        print(grid.best_score_)
        print(grid.best_estimator_.nu)
        print(grid.best_estimator_.gamma)

        self._best_model = grid.best_estimator_

        if not os.path.exists(self._target_base_directory + 'best_models/' + target_directory):
            os.makedirs(self._target_base_directory + 'best_models/' + target_directory)

        # save the model to disk
        filename = self._target_base_directory + 'best_models/' + target_directory + '/svm_rbf_model_for_' + self._antibiotic_name + '.sav'
        joblib.dump(self._best_model, filename)

    def predict_ar(self, x):
        self._best_model.predict(x)

    def test_model(self):
        y_pred = self._best_model.predict(self._x_te)

        cm = confusion_matrix(self._y_te, y_pred)

        sensitivity = float(cm[0][0]) / np.sum(cm[0])
        specificity = float(cm[1][1]) / np.sum(cm[1])

        print('For ' + self._antibiotic_name)
        print(collections.Counter(self._y_te))
        print('Sensitivity: ' + str(sensitivity))
        print('Specificity: ' + str(specificity))

        # Plot non-normalized confusion matrix
        # plot_confusion_matrix(self._y_te, y_pred, classes=['susceptible', 'resistant'], title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plot_confusion_matrix(self._y_te, y_pred, classes=['susceptible', 'resistant'], normalize=True, title='Normalized confusion matrix')

        target_directory = self._scoring + '_' + self._label_tags + '_' + self._feature_selection

        if not os.path.exists(self._target_base_directory + 'confusion_matrices/' + target_directory):
            os.makedirs(self._target_base_directory + 'confusion_matrices/' + target_directory)

        plt.savefig(self._target_base_directory + 'confusion_matrices/' + target_directory + '/normalized_svm_with_rbf_' + self._antibiotic_name + '.png')

        plot_confusion_matrix(self._y_te, y_pred, classes=['susceptible', 'resistant'], normalize=False, title='Confusion matrix')

        plt.savefig(self._target_base_directory + 'confusion_matrices/' + target_directory + '/svm_with_rbf_' + self._antibiotic_name + '.png')
