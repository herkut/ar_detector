import collections
import json
import os

import torch
import torch.utils.data as utils
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

from config import Config
from models.base_ar_detector import BaseARDetector
from models.pytorch_models.early_stopping import EarlyStopping
from utils.confusion_matrix_drawer import classification_report, concatenate_classification_reports, plot_confusion_matrix
from utils.helper_functions import get_k_fold_validation_indices, create_hyperparameter_space_for_dnn
from utils.statistical_tests.statistical_tests import choose_best_hyperparameters
import numpy as np


class FeetForwardNetwork(torch.nn.Module):
    def __init__(self, feature_size, output_size, hidden_units, activation_functions, dropout_rate, batch_normalization=False):
        """

        :param feature_size: input shape
        :param output_size: output size
        :param hidden_units: array containing hidden neuron counts in each hidden layers
        :param activation_functions: array containing activation function in each hidden layers
        :param dropout_rate: dropout rate would be applied on whole network
        :param batch_normalization: variable to enable or disable batch normalization
        """
        super(FeetForwardNetwork, self).__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.do_bn = batch_normalization
        if dropout_rate > 0:
            self.do_dropout = True
        else:
            self.do_dropout = False
        # fully connected layers
        self.fcs = []
        # batch normalizations
        self.bns = []
        # dropouts
        self.dos =[]

        self.bn_input = torch.nn.BatchNorm1d(self.feature_size, momentum=0.5)

        if len(hidden_units) != len(activation_functions):
            # TODO custom network exception
            raise Exception

        for i in range(0, len(hidden_units)):
            if i == 0:
                fc = torch.nn.Linear(self.feature_size, hidden_units[0])
                setattr(self, 'fc%i' % i, fc)
                self.fcs.append(fc)
            else:
                fc = torch.nn.Linear(hidden_units[i-1], hidden_units[i])
                setattr(self, 'fc%i' % i, fc)
                self.fcs.append(fc)

            if self.do_bn:
                bn = torch.nn.BatchNorm1d(hidden_units[i], momentum=0.5)
                setattr(self, 'bn%i' % i, bn)
                self.bns.append(bn)
            # It is required to define dropouts in model init method to make it
            # reponsible to model.eval() and model.train() methods
            if self.do_dropout:
                do = torch.nn.Dropout(p=dropout_rate)
                setattr(self, 'do%i' % i, do)
                self.dos.append(do)

        self.predict = torch.nn.Linear(hidden_units[-1], self.output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

        for af in activation_functions:
            if af not in ['relu', 'tanh', 'hardtanh', 'leaky_relu']:
                # TODO define custom not implemented activation function exception
                raise Exception('Not implemented activation function: ' + af)
        self.activations = activation_functions

    def forward(self, x):
        """

        :param x:
        :return:
        """
        if self.do_bn:
            x = self.bn_input(x)

        for i in range(0, len(self.fcs)):
            x = self.fcs[i](x)

            # Set activations
            if self.activations[i] == 'relu':
                x = torch.nn.ReLU()(x)
            elif self.activations[i] == 'tanh':
                x = torch.nn.Tanh()(x)
            elif self.activations[i] == 'hardtanh':
                x = torch.nn.Hardtanh()(x)
            elif self.activations[i] == 'leaky_relu':
                x = torch.nn.LeakyReLU()(x)

            # This contested whether batch normalization is better after activation or not
            if self.do_bn:
                x = self.bns[i](x)

            if self.do_dropout:
                x = self.dos[i](x)

        out = self.predict(x)
        out = self.softmax(out + 1e-10)
        return out

    def get_name(self):
        return 'dnn-' + str(len(self.fcs)) + 'd'


def prepare_dataloader(batch_size, x, y):
    # numpy matrix -> torch tensors
    x_tensor = torch.from_numpy(x).float()
    # numpy array -> numpy matrix (len, 1)
    # y_tr[validation_indices].reshape(y_tr[validation_indices].shape[0], -1)
    y_tensor = torch.from_numpy(y).long()

    # convert labels into one hot encoded
    # y_tr_one_hot = torch.nn.functional.one_hot(y_tr_tensor.to(torch.long), num_classes=2)
    # y_val_one_hot = torch.nn.functional.one_hot(y_val_tensor.to(torch.long), num_classes=2)

    # create dataset and dataloader
    ar_dataset = utils.TensorDataset(x_tensor, y_tensor)
    ar_dataloader = utils.DataLoader(ar_dataset, batch_size=batch_size)

    return ar_dataloader


class ARDetectorDNN(BaseARDetector):
    def __init__(self, feature_selection, feature_size=None, antibiotic_name=None, model_name='dnn', class_weights=None):
        self._batch_size = 64
        self._results_directory = Config.results_directory
        self._feature_selection = feature_selection
        self._feature_size = feature_size
        self._label_tags = Config.label_tags

        self._model = None
        self._best_model = None

        self._antibiotic_name = antibiotic_name
        self._scoring = Config.deep_learning_metric
        self._label_tags = Config.label_tags

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._target_directory = None
        self._model_name = model_name
        self._class_weights = class_weights[:, 1]
        self._target_directory = self._model_name + '_' + self._scoring + '_' + self._label_tags + '_' + self._feature_selection

    def save_model(self):
        pass

    def load_model(self):
        with open(os.path.join(Config.results_directory,
                               'best_models',
                               self._target_directory,
                               self._model_name + '_' + self._antibiotic_name + '.json')) as fp:
            best_hyperparameters = json.load(fp)

        model = FeetForwardNetwork(self._feature_size,
                                   2,
                                   best_hyperparameters['hidden_units'],
                                   best_hyperparameters['activation_functions'],
                                   best_hyperparameters['dropout_rate'],
                                   batch_normalization=True)
        model.load_state_dict(torch.load(os.path.join(Config.results_directory,
                                                      'best_models',
                                                      self._target_directory,
                                                      self._model_name + '_for_' + self._antibiotic_name + '.pt')))
        model.to(self._device)
        model.eval()

        self._best_model = model

    def tune_hyperparameters(self, param_grid, x_tr, y_tr):
        # TODO convert batch size configurative
        batch_size = 64
        # x_tr[0] sample counts x_tr[1] feature size

        # grid search
        hyperparameter_space = create_hyperparameter_space_for_dnn(param_grid)

        cv_results = {'grids': [], 'training_results': [], 'validation_results': []}
        for grid in hyperparameter_space:
            cv_results['grids'].append(grid)
            cv_result = {'training_results': [], 'validation_results': []}

            # prepare data for cross validation
            k_fold_indices = get_k_fold_validation_indices(10, x_tr, y_tr)
            print('Grid: ' + str(grid))
            for train_indeces, validation_indeces in k_fold_indices:
                # initialization of dataloaders
                ar_dataloader_tr = prepare_dataloader(batch_size, x_tr[train_indeces], y_tr[train_indeces])
                ar_dataloader_val = prepare_dataloader(batch_size, x_tr[validation_indeces], y_tr[validation_indeces])

                feature_size = self._feature_size

                # Train model with hyperparameters in the grid
                self._train_model(grid, ar_dataloader_tr, ar_dataloader_val, 'tuning_hyperparameters')

                # Training has been completed, get training and validation classification report
                # load best model and validate with training and validation test set
                model = FeetForwardNetwork(self._feature_size,
                                           2,
                                           grid['hidden_units'],
                                           grid['activation_functions'],
                                           grid['dropout_rate'],
                                           batch_normalization=True)
                model.load_state_dict(torch.load(os.path.join(Config.results_directory,
                                                              'checkpoints',
                                                              self._target_directory,
                                                              self._model_name + '_for_' + self._antibiotic_name + '.pt')))
                model.to(self._device)

                cv_result['training_results'].append(self._validate_model(model, ar_dataloader_tr, calculate_loss=False))
                cv_result['validation_results'].append(self._validate_model(model, ar_dataloader_val, calculate_loss=False))
            cv_results['training_results'].append(cv_result['training_results'])
            cv_results['validation_results'].append(cv_result['validation_results'])

        if not os.path.exists(os.path.join(Config.results_directory, 'grid_search_scores', self._target_directory)):
            os.makedirs(os.path.join(Config.results_directory, 'grid_search_scores', self._target_directory))

        with open(os.path.join(Config.results_directory,
                               'grid_search_scores',
                               self._target_directory,
                               self._model_name + '_' + self._antibiotic_name + '.json'), 'w') as fp:
            json.dump(cv_results, fp)

        # find best hyperparameters and save them in a file
        best_hyperparameters = choose_best_hyperparameters(cv_results)

        if not os.path.exists(os.path.join(Config.results_directory, 'best_models', self._target_directory)):
            os.makedirs(os.path.join(Config.results_directory, 'best_models', self._target_directory))

        with open(os.path.join(Config.results_directory,
                               'best_models',
                               self._target_directory,
                               self._model_name + '_' + self._antibiotic_name + '.json'), 'w') as fp:
            json.dump(best_hyperparameters, fp)

    def predict_ar(self, x):
        self._best_model.eval()
        return self._best_model(torch.from_numpy(x).float())

    def test_model(self, x_te, y_te):
        self._best_model.eval()
        y_pred = torch.argmax(self._best_model.forward(torch.from_numpy(x_te).float()), dim=1)
        y_pred = y_pred.numpy()

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

        plot_confusion_matrix(y_te,
                              y_pred,
                              classes=['susceptible', 'resistant'],
                              normalize=True,
                              title='Normalized confusion matrix')

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

    def _training_step(self, model, optimizer, criterion, dataloader):
        # To activate dropouts
        model.train()

        training_results = None
        training_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)
            # initialization of gradients
            optimizer.zero_grad()
            # Forward propagation
            y_hat = model(inputs)
            pred = torch.argmax(y_hat, dim=1)
            # Computation of cost function
            cost = criterion(y_hat, labels)
            # Back propagation
            cost.backward()
            # Update parameters
            optimizer.step()

            training_loss += cost

            tmp_classification_report = classification_report(labels, pred)
            if training_results is None:
                training_results = tmp_classification_report
            else:
                training_results = concatenate_classification_reports(training_results,
                                                                      tmp_classification_report)
        training_results['loss'] = training_loss

        return training_results

    def _validate_model(self, model, dataloader, criterion=None, calculate_loss=False):
        # To deactivate dropouts
        model.eval()

        validation_results = None

        if calculate_loss:
            validation_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)

            y_hat = model(inputs)
            pred = torch.argmax(y_hat, dim=1)

            if calculate_loss:
                cost = criterion(y_hat, labels)
                validation_loss += cost

            tmp_classification_report = classification_report(labels, pred)
            if validation_results is None:
                validation_results = tmp_classification_report
            else:
                validation_results = concatenate_classification_reports(validation_results,
                                                                        tmp_classification_report)

        if calculate_loss:
            validation_results['loss'] = validation_loss

        return validation_results

    def _train_model(self, hyperparameters, ar_dataloader_tr, ar_dataloader_val, training_purpose):
        feature_size = self._feature_size

        model, criterion, optimizer = self._initialize_model(feature_size, hyperparameters)

        if training_purpose == 'tuning_hyperparameters':
            # Create the directory holding checkpoints if not exists
            if not os.path.exists(os.path.join(Config.results_directory, 'checkpoints')):
                os.makedirs(os.path.join(Config.results_directory, 'checkpoints'))

            if not os.path.exists(os.path.join(Config.results_directory, 'checkpoints', self._target_directory)):
                os.makedirs(os.path.join(Config.results_directory, 'checkpoints', self._target_directory))

            es = EarlyStopping(metric=self._scoring,
                               mode='max' if self._scoring in ['f1', 'accuracy', 'sensitivity/recall', 'precision'] else 'min',
                               patience=10,
                               checkpoint_file=os.path.join(Config.results_directory,
                                                            'checkpoints',
                                                            self._target_directory,
                                                            self._model_name + '_for_' + self._antibiotic_name + '.pt'))
        elif training_purpose == 'training_best_model':
            # Create the directory holding best models if not exists
            if not os.path.exists(os.path.join(Config.results_directory, 'best_models')):
                os.makedirs(os.path.join(Config.results_directory, 'best_models'))

            if not os.path.exists(os.path.join(Config.results_directory, 'best_models', self._target_directory)):
                os.makedirs(os.path.join(Config.results_directory, 'best_models', self._target_directory))

            # Save checkpoint as best model
            es = EarlyStopping(metric=self._scoring,
                               mode='max' if self._scoring in ['f1', 'accuracy', 'sensitivity/recall', 'precision'] else 'min',
                               patience=10,
                               checkpoint_file=os.path.join(Config.results_directory,
                                                            'best_models',
                                                            self._target_directory,
                                                            self._model_name + '_for_' + self._antibiotic_name + '.pt'))
        else:
            raise Exception('Unknown training purpose')

        for epoch in range(2000):
            # Training
            self._training_step(model, optimizer, criterion, ar_dataloader_tr)

            # Validation
            training_results = self._validate_model(model, ar_dataloader_tr, criterion=criterion, calculate_loss=True)
            validation_results = self._validate_model(model, ar_dataloader_val, criterion=criterion, calculate_loss=True)

            if es.step(epoch, validation_results, model):
                # print('Early stopping at epoch: ' + str(epoch) + ' best index: ' + str(es.best_index))
                print('Best epoch: ' + str(es.best_index) + ', best metrics: ' + str(es.best_metrics))
                break

            tr_str = '[%d] training loss: %.9f, ' + self._scoring + ': %.9f'
            val_str = '[%d] validation loss: %.9f, ' + self._scoring + ': %.9f'
            print(tr_str % (epoch, training_results['loss'], training_results[self._scoring]))
            print(val_str % (epoch, validation_results['loss'], validation_results[self._scoring]))

    def train_best_model(self, hyperparameters, x_tr, y_tr, x_te, y_te):
        batch_size = 64
        ar_dataloader_tr = prepare_dataloader(batch_size, x_tr, y_tr)
        ar_dataloader_te = prepare_dataloader(batch_size, x_te, y_te)

        self._train_model(hyperparameters, ar_dataloader_tr, ar_dataloader_te, 'training_best_model')

        self.load_model()

    def _initialize_model(self, feature_size, hyperparameters):
        # initialization of the model
        model = FeetForwardNetwork(feature_size,
                                   2,
                                   hyperparameters['hidden_units'],
                                   hyperparameters['activation_functions'],
                                   hyperparameters['dropout_rate'],
                                   batch_normalization=True)
        # put model into gpu if exists
        model.to(self._device)
        # initialization completed
        # Optimizer initialization
        # class weight is used to handle unbalanced data
        # BCEWithLogitsLoss = Sigmoid->BCELoss
        # CrossEntropyLoss = LogSoftmax->NLLLoss
        criterion = torch.nn.NLLLoss(reduction='mean', weight=torch.from_numpy(self._class_weights).to(self._device))
        if hyperparameters['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
        elif hyperparameters['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=hyperparameters['learning_rate'])
        elif hyperparameters['optimizer'] == 'Adamax':
            optimizer = torch.optim.Adamax(model.parameters(), lr=hyperparameters['learning_rate'])
        elif hyperparameters['optimizer'] == 'RMSProp':
            optimizer = torch.optim.RMSProp(model.parameters(), lr=hyperparameters['learning_rate'])
        else:
            raise Exception('Not implemented optimizer: ' + hyperparameters['optimizer'])
        # Optimizer initialization completed
        return model, criterion, optimizer
