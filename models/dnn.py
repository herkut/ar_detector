import collections
import json
import os

from keras import backend as K
from keras import Sequential, callbacks
from keras.engine.saving import model_from_json
from keras.layers import Dense, BatchNormalization, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils.confusion_matrix_drawer import plot_confusion_matrix
from utils.numpy_encoder import NumpyEncoder


class ArDetectorByDNN:
    def __init__(self, results_directory, feature_selection, antibiotic_name, feature_size, label_tags='phenotype', metrics=['accuracy']):
        self._x_tr = None
        self._y_tr = None
        self._x_te = None
        self._y_te = None
        self._results_directory = results_directory
        self._feature_selection = feature_selection
        self._antibiotic_name = antibiotic_name
        self._feature_size = feature_size
        self._label_tags = label_tags

        self._metrics = metrics

        self._best_model = None
        self._target_directory = 'dnn_' + self._metrics[0] + '_' + self._label_tags + '_' + self._feature_selection

    def create_model(self, hidden_units=[16], activation_functions=['relu'], batch_normalization_required=False, optimizer='adam', dropout_rate=0.0):
        # Clear tensorflow graphs to avoid OOM
        if K.backend() == 'tensorflow':
            K.clear_session()

        model = Sequential()
        self._target_directory = 'dnn_' + self._metrics[0] + '_' + self._label_tags + '_' + self._feature_selection

        # Initialize first hidden layer
        model.add(Dense(hidden_units[0], input_dim=self._feature_size, activation=activation_functions[0]))
        if batch_normalization_required:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        self._target_directory = self._target_directory + '_' + str(hidden_units[0]) + '-' + activation_functions[0]

        for i in range(1, len(hidden_units)):
            model.add(Dense(hidden_units[i], activation=activation_functions[i]))
            if batch_normalization_required:
                model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            self._target_directory = self._target_directory + '_' + str(hidden_units[i]) + '-' + activation_functions[i]
        # Binary classification would be done for each antibiotic separately
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=self._metrics)

        return model

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

    def tune_hyperparameters(self, param_grid):
        model = KerasClassifier(build_fn=self.create_model, verbose=1)

        es = callbacks.EarlyStopping(monitor='loss',
                                     min_delta=0,
                                     patience=10,
                                     verbose=0,
                                     mode='auto')

        kfold = StratifiedKFold(n_splits=5, random_state=0)
        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            cv=kfold,
                            n_jobs=1)

        grid_result = grid.fit(self._x_tr, self._y_tr, callbacks=[es])
        #grid_result = grid.fit(self._x_tr, self._y_tr)

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        if not os.path.exists(self._results_directory + 'best_models/' + self._target_directory):
            os.makedirs(self._results_directory + 'best_models/' + self._target_directory)

        if not os.path.exists(self._results_directory + 'grid_search_scores/' + self._target_directory):
            os.makedirs(self._results_directory + 'grid_search_scores/' + self._target_directory)

        with open(self._results_directory + 'grid_search_scores/' + self._target_directory + '/dnn_' + self._antibiotic_name + '.json', 'w') as f:
            f.write(json.dumps(grid.cv_results_, cls=NumpyEncoder))

        print(grid_result.best_score_)
        print(grid_result.best_params_)

        self._best_model = grid.best_estimator_.model

        self.save_model(grid.best_estimator_.model)

        if K.backend() == 'tensorflow':
            K.clear_session()

    def save_model(self, model):
        if not os.path.exists(self._results_directory + 'best_models/' + self._target_directory):
            os.makedirs(self._results_directory + 'best_models/' + self._target_directory)

        # save the models into file
        model_json = model.to_json()
        with open(self._results_directory + 'best_models/' + self._target_directory + '/model_' + self._antibiotic_name + '.json', "w") as json_file:
            json_file.write(model_json)
        # save network weights
        model.save_weights(self._results_directory + 'best_models/' + self._target_directory + "/dnn_for_" + self._antibiotic_name + '.h5')

    def load_model(self):
        # Model reconstruction from JSON file
        with open(self._results_directory + 'best_models/' + self._target_directory + '/model_' + self._antibiotic_name + '.json', 'r') as f:
            self._best_model = model_from_json(f.read())

        # Load weights into the new model
        self._best_model.load_weights(self._results_directory + 'best_models/' + self._target_directory + "/dnn_for_" + self._antibiotic_name + '.h5')

    def test_model(self):
        y_pred = self._best_model.predict(self._x_te)

        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

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

        # Plot non-normalized confusion matrix
        # plot_confusion_matrix(self._y_te, y_pred, classes=['susceptible', 'resistant'], title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plot_confusion_matrix(self._y_te, y_pred, classes=['susceptible', 'resistant'], normalize=True, title='Normalized confusion matrix')

        if not os.path.exists(self._results_directory + 'confusion_matrices/' + self._target_directory):
            os.makedirs(self._results_directory + 'confusion_matrices/' + self._target_directory)

        plt.savefig(self._results_directory + 'confusion_matrices/' + self._target_directory + '/normalized_dnn_' + self._antibiotic_name + '.png')

        plot_confusion_matrix(self._y_te, y_pred, classes=['susceptible', 'resistant'], normalize=False, title='Confusion matrix')

        plt.savefig(self._results_directory + 'confusion_matrices/' + self._target_directory + '/dnn_' + self._antibiotic_name + '.png')

        y_true = pd.Series(self._y_te, name="Actual")
        y_pred = pd.Series(y_pred[:, 0], name="Predicted")
        df_confusion = pd.crosstab(y_true, y_pred)
        df_confusion.to_csv(self._results_directory + 'confusion_matrices/' + self._target_directory + '/dnn_' + self._antibiotic_name + '.csv')

        if K.backend() == 'tensorflow':
            K.clear_session()