import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib as plt

from utils.confusion_matrix_drawer import plot_confusion_matrix
from utils.numpy_encoder import NumpyEncoder


class ARDetectorByRandomForest:
    def __init__(self, antibiotic_name, label_tags='phenotype', scoring='roc_auc'):
        self._x_tr = None
        self._y_tr = None
        self._x_te = None
        self._y_te = None
        self._label_tags = label_tags
        self._model = RandomForestClassifier()
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
        self._best_model = joblib.load('/home/herkut/Desktop/ar_detector/best_models/' + self._scoring + '_' + self._label_tags + '/random_forest_model_for_' + self._antibiotic_name + '.sav')

    def tune_hyperparameters(self, n_estimators,  max_features, bootstrap=None, max_depth=None):
        param_grid= {'n_estimators': n_estimators, 'max_features': max_features}

        if bootstrap is not None:
            param_grid['bootstrap'] = bootstrap

        if max_depth is not None:
            param_grid['max_depth'] = max_depth

        model = self._model

        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=self._scoring, cv=5, verbose=True, n_jobs=-1)
        grid.fit(self._x_tr, self._y_tr)

        print(grid)

        if not os.path.exists('/home/herkut/Desktop/ar_detector/grid_search_scores/' + self._scoring + '_' + self._label_tags):
            os.makedirs('/home/herkut/Desktop/ar_detector/grid_search_scores/' + self._scoring + '_' + self._label_tags)

        with open('/home/herkut/Desktop/ar_detector/grid_search_scores/' + self._scoring + '_' + self._label_tags + '/random_forest_' + self._antibiotic_name + '.json', 'w') as f:
            f.write(json.dumps(grid.cv_results_, cls=NumpyEncoder))

        # summarize the results of the grid search
        print('Summary of the model:')
        print(grid.best_score_)
        print(grid.best_estimator_.bootstrap)
        print(grid.best_estimator_.n_estimators)
        print(grid.best_estimator_.max_depth)
        print(grid.best_estimator_.max_features)

        self._best_model = grid.best_estimator_

        if not os.path.exists('/home/herkut/Desktop/ar_detector/best_models/' + self._scoring + '_' + self._label_tags):
            os.makedirs('/home/herkut/Desktop/ar_detector/best_models/' + self._scoring + '_' + self._label_tags)

        # save the model to disk
        filename = '/home/herkut/Desktop/ar_detector/best_models/' + self._scoring + '_' + self._label_tags + '/random_forest_model_for_' + self._antibiotic_name + '.sav'
        joblib.dump(self._best_model, filename)

    def predict_ar(self, x):
        self._best_model.predict(x)

    def test_model(self):
        y_pred = self._best_model.predict(self._x_te)

        # Plot non-normalized confusion matrix
        # plot_confusion_matrix(self._y_te, y_pred, classes=['susceptible', 'resistant'], title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plot_confusion_matrix(self._y_te, y_pred, classes=['susceptible', 'resistant'], normalize=True, title='Normalized confusion matrix')

        if not os.path.exists('/home/herkut/Desktop/ar_detector/confusion_matrices/' + self._scoring + '_' + self._label_tags):
            os.makedirs('/home/herkut/Desktop/ar_detector/confusion_matrices/' + self._scoring + '_' + self._label_tags)

        plt.savefig('/home/herkut/Desktop/ar_detector/confusion_matrices/' + self._scoring + '_' + self._label_tags + '/random_forest_' + self._antibiotic_name + '.png')
