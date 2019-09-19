import json
import os

import torch
import torch.utils.data as utils

import numpy as np
from torch.nn import BCELoss

from config import Config
from models.base_ar_detector import BaseARDetector
from utils.confusion_matrix_drawer import classification_report, concatenate_classification_reports
from utils.helper_functions import get_k_fold_validation_indices, create_hyperparameter_space
from utils.statistical_tests.statistical_tests import choose_best_hyperparameters


class EarlyStopping(object):
    def __init__(self, metric='loss', mode='min', min_delta=0, patience=10, checkpoint_file='/tmp/dnn_checkpoint.pt'):
        """

        :param metric: loss, accuracy, f1, sensitivity, specificity, precision
        :param mode:
        :param min_delta:
        :param patience: how many bad epochs is required to stop early
        """
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.bad_epoch_count = 0
        self.best_index = None
        self.best_metrics = None
        self.best_model = None

        self.checkpoint_file = checkpoint_file

    def step(self, epoch, results, model):
        # From Goodfellow's Deep Learning book
        if self.best_index is None:
            self.best_index = epoch
            self.best_metrics = results
        else:
            if self.mode == 'min':
                if self.best_metrics[self.metric] - results[self.metric] > self.min_delta:
                    # Update best metrics and save checkpoint
                    self.save_checkpoint(epoch, results, model)
                else:
                    self.bad_epoch_count += 1
            else:
                if self.best_metrics[self.metric] - results[self.metric] < self.min_delta:
                    # Update best metrics and save checkpoint
                    self.save_checkpoint(epoch, results, model)
                else:
                    self.bad_epoch_count += 1
            if self.bad_epoch_count > self.patience:
                return True
            else:
                return False

    def save_checkpoint(self, epoch, results, model):
        self.best_index = epoch
        self.best_metrics = results
        self.bad_epoch_count = 0
        torch.save(model.state_dict(), self.checkpoint_file)


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

            # This contested whether batch normalization is better after actiation or not
            if self.do_bn:
                x = self.bns[i](x)

            if self.do_dropout:
                x = self.dos[i](x)

        out = self.predict(x)
        out = self.softmax(out)
        return out

    def get_name(self):
        return 'dnn-' + str(len(self.fcs)) + 'd'


def prepare_dataloaders(batch_size, x_tr, y_tr, train_indices, validation_indices):
    # numpy matrix -> torch tensors
    x_tr_tensor = torch.from_numpy(x_tr[train_indices]).float()
    # numpy array -> numpy matrix (len, 1)
    # y_tr[validation_indices].reshape(y_tr[validation_indices].shape[0], -1)
    y_tr_tensor = torch.from_numpy(y_tr[train_indices]).long()

    # numpy matrix -> torch tensors
    x_val_tensor = torch.from_numpy(x_tr[validation_indices]).float()
    # numpy array -> numpy matrix (len, 1)
    # y_tr[validation_indices].reshape(y_tr[validation_indices].shape[0], -1)
    y_val_tensor = torch.from_numpy(y_tr[validation_indices]).long()

    # convert labels into one hot encoded
    y_tr_one_hot = torch.nn.functional.one_hot(y_tr_tensor.to(torch.long), num_classes=2)
    y_val_one_hot = torch.nn.functional.one_hot(y_val_tensor.to(torch.long), num_classes=2)

    # create dataset and dataloader
    ar_dataset_tr = utils.TensorDataset(x_tr_tensor, y_tr_tensor)
    ar_dataloader_tr = utils.DataLoader(ar_dataset_tr, batch_size=batch_size)
    ar_dataset_val = utils.TensorDataset(x_val_tensor, y_val_tensor)
    ar_dataloader_val = utils.DataLoader(ar_dataset_val, batch_size=batch_size)

    return ar_dataloader_tr, ar_dataloader_val


class ARDetectorDNN(BaseARDetector):
    def __init__(self, feature_selection, antibiotic_name=None, model_name='dnn', class_weights=None):
        self._results_directory = Config.results_directory
        self._feature_selection = feature_selection
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
        pass

    def tune_hyperparameters(self, param_grid, x_tr, y_tr):
        batch_size = 64
        # x_tr[0] sample counts x_tr[1] feature size
        feature_size = x_tr.shape[1]

        # grid search
        hyperparameter_space = create_hyperparameter_space(param_grid)

        cv_results = {'grids': [], 'training_results': [], 'validation_results': []}
        for grid in hyperparameter_space:
            cv_results['grids'].append(grid)
            cv_result = {'training_results': [], 'validation_results': []}

            # prepare data for closs validation
            k_fold_indices = get_k_fold_validation_indices(10, x_tr, y_tr)
            print('Grid: ' + str(grid))
            for train_indeces, validation_indeces in k_fold_indices:
                # initialization of dataloaders
                ar_dataloader_tr, ar_dataloader_val = prepare_dataloaders(batch_size, x_tr, y_tr, train_indeces, validation_indeces)

                # initialization of the model
                model = FeetForwardNetwork(feature_size,
                                           2,
                                           grid['hidden_units'],
                                           grid['activation_functions'],
                                           grid['dropout_rate'],
                                           batch_normalization=True)

                # put model into gpu if exists
                model.to(self._device)
                # initialization completed

                # Optimizer initialization
                # class weight is used to handle unbalanced data
                # BCEWithLogitsLoss = Sigmoid->BCELoss
                # CrossEntropyLoss = LogSoftmax->NLLLoss
                criterion = torch.nn.NLLLoss(reduction='mean', weight=torch.from_numpy(self._class_weights).to(self._device))
                if grid['optimizer'] == 'Adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=grid['learning_rate'])
                elif grid['optimizer'] == 'SGD':
                    optimizer = torch.optim.SGD(model.parameters(), lr=grid['learning_rate'])
                elif grid['optimizer'] == 'Adamax':
                    optimizer = torch.optim.Adamax(model.parameters(), lr=grid['learning_rate'])
                elif grid['optimizer'] == 'RMSProp':
                    optimizer = torch.optim.RMSProp(model.parameters(), lr=grid['learning_rate'])
                else:
                    raise Exception('Not implemented optimizer: ' + grid['optimizer'])
                # Optimizer initialization completed

                # create the directory holding checkpoints if not exists
                if not os.path.exists(os.path.join(Config.results_directory, 'checkpoints')):
                    os.makedirs(os.path.join(Config.results_directory, 'checkpoints'))

                es = EarlyStopping(metric='loss',
                                   mode='min',
                                   patience=10,
                                   checkpoint_file=os.path.join(Config.results_directory, 'checkpoints', self._model_name + '_checkpoint.pt'))

                for epoch in range(2000):
                    model.train()

                    training_results = None
                    validation_results = None

                    # Training
                    training_results = self._training_step(model, optimizer, criterion, ar_dataloader_tr)

                    # Validation
                    validation_results = self._validate_model(model, criterion, ar_dataloader_val)

                    if es.step(epoch, validation_results, model):
                        # print('Early stopping at epoch: ' + str(epoch) + ' best index: ' + str(es.best_index))
                        print('Best metrics: ' + str(es.best_metrics))
                        break

                    # print('[%d] training loss: %.9f' % (epoch + 1, training_results['loss']))
                    # print('[%d] validation loss: %.9f' % (epoch + 1, validation_results['loss']))
                # Training has been completed, get training and validation classification report
                # load best model and validate with training and validation test set
                model = FeetForwardNetwork(feature_size,
                                           2,
                                           grid['hidden_units'],
                                           grid['activation_functions'],
                                           grid['dropout_rate'],
                                           batch_normalization=True)
                model.load_state_dict(torch.load(os.path.join(Config.results_directory, 'checkpoints', self._model_name + '_checkpoint.pt')))
                model.to(self._device)
                model.eval()
                cv_result['training_results'].append(self._calculate_model_performance(model, ar_dataloader_tr))
                cv_result['validation_results'].append(self._calculate_model_performance(model, ar_dataloader_val))
            cv_results['training_results'].append(cv_result['training_results'])
            cv_results['validation_results'].append(cv_result['validation_results'])

        # TODO store cv_results json in related results directory
        if not os.path.exists(os.path.join(Config.results_directory, 'best_models', self._target_directory)):
            os.makedirs(os.path.join(Config.results_directory, 'best_models', self._target_directory))

        with open(os.path.join(Config.results_directory, 'best_models', self._target_directory, self._model_name + '.json'), 'w') as fp:
            json.dump(cv_results, fp)

    def predict_ar(self, x):
        self._model.eval()
        return self._model.forward(x)

    def test_model(self, x_te, y_te):
        self._model.eval()
        pass

    def _training_step(self, model, optimizer, criterion, dataloader):
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

    def _validate_model(self, model, criterion, dataloader):
        validation_results = None
        validation_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)

            y_hat = model(inputs)
            pred = torch.argmax(y_hat, dim=1)
            cost = criterion(y_hat, labels)
            validation_loss += cost

            tmp_classification_report = classification_report(labels, pred)
            if validation_results is None:
                validation_results = tmp_classification_report
            else:
                validation_results = concatenate_classification_reports(validation_results,
                                                                        tmp_classification_report)
        validation_results['loss'] = validation_loss

        return validation_results

    def _calculate_model_performance(self, model, dataloader):
        validation_results = None
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)

            y_hat = model(inputs)
            pred = torch.argmax(y_hat, dim=1)

            tmp_classification_report = classification_report(labels, pred)
            if validation_results is None:
                validation_results = tmp_classification_report
            else:
                validation_results = concatenate_classification_reports(validation_results,
                                                                        tmp_classification_report)
        return validation_results
