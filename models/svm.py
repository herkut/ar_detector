import collections
import json
import os

import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd

from utils.confusion_matrix_drawer import plot_confusion_matrix
from utils.numpy_encoder import NumpyEncoder

import config as cfg


class ARDetectorBySVMWithRBF:
    def __init__(self, target_base_directory, feature_selection, antibiotic_name=None, label_tags='phenotype', scoring='roc_auc', class_weights=None):
        self._target_base_directory = target_base_directory
        self._feature_selection = feature_selection
        self._label_tags = label_tags
        if class_weights is None:
            self._model = svm.SVC(kernel='rbf')
        else:
            self._model = svm.SVC(kernel='rbf', class_weight=class_weights)
        self._best_model = None
        self._antibiotic_name = antibiotic_name
        self._scoring = scoring
        self._target_directory = 'svm_' + self._scoring + '_' + self._label_tags + '_' + self._feature_selection

    def set_antibiotic_name(self, antibiotic_name):
        self._antibiotic_name = antibiotic_name

    def reinitialize_model_with_parameters(self, parameters, class_weights=None):
        if class_weights is None:
            self._model = svm.SVC(kernel='rbf', C=parameters['C'], gamma=parameters['gamma'])
        else:
            self._model = svm.SVC(kernel='rbf', C=parameters['C'], gamma=parameters['gamma'], class_weight=class_weights)

    def load_model(self):
        self._best_model = joblib.load(self._target_base_directory + 'best_models/' + self._target_directory + '/svm_rbf_model_for_' + self._antibiotic_name + '.sav')

    def tune_hyperparameters(self, param_grid, x_tr, y_tr):
        model = self._model

        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=self._scoring, cv=5, verbose=True, n_jobs=-1)
        grid.fit(x_tr, y_tr)

        print(grid)

        if not os.path.exists(self._target_base_directory + 'grid_search_scores/' + self._target_directory):
            os.makedirs(self._target_base_directory + 'grid_search_scores/' + self._target_directory)

        with open(self._target_base_directory + 'grid_search_scores/' + self._target_directory + '/svm_rbf_' + self._antibiotic_name + '.json', 'w') as f:
            f.write(json.dumps(grid.cv_results_, cls=NumpyEncoder))

        # summarize the results of the grid search
        if not os.path.exists(self._target_base_directory + 'best_models/' + self._target_directory):
            os.makedirs(self._target_base_directory + 'best_models/' + self._target_directory)

        with open(self._target_base_directory + 'best_models/' + self._target_directory + '/svm_rbf_' + self._antibiotic_name + '.json','w') as f:
            f.write(json.dumps(grid.best_params_, cls=NumpyEncoder))

        print('Summary of the model:')
        print(grid.best_score_)
        print(grid.best_estimator_.nu)
        print(grid.best_estimator_.gamma)

        self._best_model = grid.best_estimator_

        if not os.path.exists(self._target_base_directory + 'best_models/' + self._target_directory):
            os.makedirs(self._target_base_directory + 'best_models/' + self._target_directory)

        # save the model to disk
        filename = self._target_base_directory + 'best_models/' + self._target_directory + '/svm_rbf_model_for_' + self._antibiotic_name + '.sav'
        joblib.dump(self._best_model, filename)

    def predict_ar(self, x):
        self._best_model.predict(x)

    def predict(self, x):
        return self._model.predict(x)

    def train_model(self, x_tr, y_tr):
        self._model.fit(x_tr, y_tr)

    def test_model(self, x_te, y_te):
        y_pred = self._best_model.predict(x_te)

        cm = confusion_matrix(y_te, y_pred)
        if np.shape(cm)[0] == 2 and np.shape(cm)[1] == 2 :
            sensitivity = float(cm[0][0]) / np.sum(cm[0])
            specificity = float(cm[1][1]) / np.sum(cm[1])
            print('For ' + self._antibiotic_name)
            print(collections.Counter(y_te))
            print('Sensitivity: ' + str(sensitivity))
            print('Specificity: ' + str(specificity))
        else:
            print('For ' + self._antibiotic_name)
            print('There has been an error in calculating sensitivity and specificity')

        plot_confusion_matrix(y_te, y_pred, classes=['susceptible', 'resistant'], normalize=True, title='Normalized confusion matrix')

        if not os.path.exists(self._target_base_directory + 'confusion_matrices/' + self._target_directory):
            os.makedirs(self._target_base_directory + 'confusion_matrices/' + self._target_directory)

        plt.savefig(self._target_base_directory + 'confusion_matrices/' + self._target_directory + '/normalized_svm_with_rbf_' + self._antibiotic_name + '.png')

        plot_confusion_matrix(y_te, y_pred, classes=['susceptible', 'resistant'], normalize=False, title='Confusion matrix')

        plt.savefig(self._target_base_directory + 'confusion_matrices/' + self._target_directory + '/svm_with_rbf_' + self._antibiotic_name + '.png')

        y_true = pd.Series(y_te, name="Actual")
        y_pred = pd.Series(y_pred, name="Predicted")
        df_confusion = pd.crosstab(y_true, y_pred)
        df_confusion.to_csv(self._target_base_directory + 'confusion_matrices/' + self._target_directory + '/svm_rbf_' + self._antibiotic_name + '.csv')


class ARDetectorBySVMWithLinear:
    def __init__(self, target_base_directory, feature_selection, antibiotic_name=None, label_tags='phenotype', scoring='roc_auc', class_weights=None):
        self._target_base_directory = target_base_directory
        self._feature_selection = feature_selection
        self._label_tags = label_tags
        if class_weights is None:
            self._model = svm.SVC(kernel='linear')
        else:
            self._model = svm.SVC(kernel='linear', class_weight=class_weights)
        self._best_model = None
        self._antibiotic_name = antibiotic_name
        self._scoring = scoring
        self._target_directory = 'svm_' + self._scoring + '_' + self._label_tags + '_' + self._feature_selection

    def set_antibiotic_name(self, antibiotic_name):
        self._antibiotic_name = antibiotic_name

    def reinitialize_model_with_parameters(self, parameters, class_weights=None):
        if class_weights is None:
            self._model = svm.SVC(kernel='linear', C=parameters['C'], gamma=parameters['gamma'])
        else:
            self._model = svm.SVC(kernel='linear', C=parameters['C'], gamma=parameters['gamma'], class_weight=class_weights)

    def load_model(self):
        self._best_model = joblib.load(self._target_base_directory + 'best_models/' + self._target_directory + '/svm_linear_model_for_' + self._antibiotic_name + '.sav')

    def tune_hyperparameters(self, param_grid, x_tr, y_tr):
        model = self._model

        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=self._scoring, cv=5, verbose=True, n_jobs=-1)
        grid.fit(x_tr, y_tr)

        print(grid)

        if not os.path.exists(self._target_base_directory + 'grid_search_scores/' + self._target_directory):
            os.makedirs(self._target_base_directory + 'grid_search_scores/' + self._target_directory)

        with open(self._target_base_directory + 'grid_search_scores/' + self._target_directory + '/svm_linear_' + self._antibiotic_name + '.json', 'w') as f:
            f.write(json.dumps(grid.cv_results_, cls=NumpyEncoder))

        # summarize the results of the grid search
        if not os.path.exists(self._target_base_directory + 'best_models/' + self._target_directory):
            os.makedirs(self._target_base_directory + 'best_models/' + self._target_directory)

        with open(self._target_base_directory + 'best_models/' + self._target_directory + '/svm_linear_' + self._antibiotic_name + '.json','w') as f:
            f.write(json.dumps(grid.best_params_, cls=NumpyEncoder))

        print('Summary of the model:')
        print(grid.best_score_)
        print(grid.best_estimator_.nu)

        self._best_model = grid.best_estimator_

        if not os.path.exists(self._target_base_directory + 'best_models/' + self._target_directory):
            os.makedirs(self._target_base_directory + 'best_models/' + self._target_directory)

        # save the model to disk
        filename = self._target_base_directory + 'best_models/' + self._target_directory + '/svm_linear_model_for_' + self._antibiotic_name + '.sav'
        joblib.dump(self._best_model, filename)

    def predict_ar(self, x):
        self._best_model.predict(x)

    def predict(self, x):
        return self._model.predict(x)

    def train_model(self, x_tr, y_tr):
        self._model.fit(x_tr, y_tr)

    def test_model(self, x_te, y_te):
        y_pred = self._best_model.predict(x_te)

        cm = confusion_matrix(y_te, y_pred)
        if np.shape(cm)[0] == 2 and np.shape(cm)[1] == 2:
            sensitivity = float(cm[0][0]) / np.sum(cm[0])
            specificity = float(cm[1][1]) / np.sum(cm[1])
            print('For ' + self._antibiotic_name)
            print(collections.Counter(y_te))
            print('Sensitivity: ' + str(sensitivity))
            print('Specificity: ' + str(specificity))
        else:
            print('For ' + self._antibiotic_name)
            print('There has been an error in calculating sensitivity and specificity')

        plot_confusion_matrix(y_te, y_pred, classes=['susceptible', 'resistant'], normalize=True, title='Normalized confusion matrix')

        if not os.path.exists(self._target_base_directory + 'confusion_matrices/' + self._target_directory):
            os.makedirs(self._target_base_directory + 'confusion_matrices/' + self._target_directory)

        plt.savefig(self._target_base_directory + 'confusion_matrices/' + self._target_directory + '/normalized_svm_with_linear_' + self._antibiotic_name + '.png')

        plot_confusion_matrix(y_te, y_pred, classes=['susceptible', 'resistant'], normalize=False, title='Confusion matrix')

        plt.savefig(self._target_base_directory + 'confusion_matrices/' + self._target_directory + '/svm_with_linear_' + self._antibiotic_name + '.png')

        y_true = pd.Series(y_te, name="Actual")
        y_pred = pd.Series(y_pred, name="Predicted")
        df_confusion = pd.crosstab(y_true, y_pred)
        df_confusion.to_csv(self._target_base_directory + 'confusion_matrices/' + self._target_directory + '/svm_linear_' + self._antibiotic_name + '.csv')
