import collections
import json
import os

import torch
import torch.utils.data as utils
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from models.base_ar_detector import BaseARDetector
from models.pytorch_models.early_stopping import EarlyStopping
from utils.confusion_matrix_drawer import classification_report, concatenate_classification_reports, plot_confusion_matrix
from utils.helper_functions import get_k_fold_indices, create_hyperparameter_space
from utils.statistical_tests.statistical_tests import choose_best_hyperparameters


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
    def __init__(self, feature_selection, dataset, feature_size=None, antibiotic_name=None, model_name='dnn', class_weights=None):
        self._results_directory = Config.results_directory
        self._feature_selection = feature_selection
        self._dataset = dataset
        self._results_directory = self._results_directory + '_' + self._dataset
        self._feature_size = feature_size
        self._label_tags = Config.label_tags

        self._batch_size = 64

        self._model = None
        self._best_model = None
        self._antibiotic_name = antibiotic_name
        self._scoring = Config.deep_learning_metric
        self._label_tags = Config.label_tags

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._target_directory = None
        self._model_name = model_name
        if class_weights is not None:
            self._class_weights = class_weights[:, 1]
        else:
            self._class_weights = None
        self._target_directory = self._model_name + '_' + self._scoring + '_' + self._label_tags + '_' + self._feature_selection

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
        if self._class_weights is not None:
            criterion = torch.nn.NLLLoss(reduction='mean', weight=torch.from_numpy(self._class_weights).to(self._device))
        else:
            criterion = torch.nn.NLLLoss(reduction='mean')

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

    def save_model(self):
        pass

    def load_model(self):
        with open(os.path.join(self._results_directory,
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
        model.load_state_dict(torch.load(os.path.join(self._results_directory,
                                                      'best_models',
                                                      self._target_directory,
                                                      self._model_name + '_' + self._antibiotic_name + '.pt')))
        model.to(self._device)
        model.eval()

        self._best_model = model

    def tune_hyperparameters(self, param_grid, x_tr, y_tr):
        # x_tr[0] sample counts x_tr[1] feature size
        feature_size = x_tr.shape[1]

        # grid search
        hyperparameter_space = create_hyperparameter_space(param_grid)

        cv_results = {'grids': [], 'training_results': [], 'validation_results': []}
        for grid in hyperparameter_space:
            batch_size = grid['batch_size']
            cv_results['grids'].append(grid)
            cv_result = {'training_results': [], 'validation_results': []}

            # prepare data for closs validation
            k_fold_indices = get_k_fold_indices(10, x_tr, y_tr)
            print('Grid: ' + str(grid))
            for train_indeces, validation_indeces in k_fold_indices:
                # initialization of dataloaders
                dataloader_tr = prepare_dataloader(batch_size, x_tr[train_indeces], y_tr[train_indeces])
                dataloader_val = prepare_dataloader(batch_size, x_tr[validation_indeces], y_tr[validation_indeces])

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
                if not os.path.exists(os.path.join(self._results_directory, 'checkpoints')):
                    os.makedirs(os.path.join(self._results_directory, 'checkpoints'))

                es = EarlyStopping(metric='loss',
                                   mode='min',
                                   patience=10,
                                   checkpoint_file=os.path.join(self._results_directory, 'checkpoints', self._model_name + '_checkpoint.pt'))

                for epoch in range(200):
                    model.train()

                    training_results = None
                    validation_results = None

                    # Training
                    training_results = self._training_step(model, optimizer, criterion, dataloader_tr)

                    # Validation
                    validation_results = self._validate_model(model, criterion, dataloader_val)

                    # print('[%d] training loss: %.9f' % (epoch, training_results['loss']))
                    # print('[%d] validation loss: %.9f' % (epoch, validation_results['loss']))

                    if es.step(epoch, validation_results, model):
                        # print('Early stopping at epoch: ' + str(epoch) + ' best index: ' + str(es.best_index))
                        print('Epoch: ' + str(es.best_index) + ', best metrics: ' + str(es.best_metrics))
                        break

                # Training has been completed, get training and validation classification report
                # load best model and validate with training and validation test set
                model = FeetForwardNetwork(feature_size,
                                           2,
                                           grid['hidden_units'],
                                           grid['activation_functions'],
                                           grid['dropout_rate'],
                                           batch_normalization=True)
                model.load_state_dict(torch.load(os.path.join(self._results_directory, 'checkpoints', self._model_name + '_checkpoint.pt')))
                model.to(self._device)
                model.eval()
                cv_result['training_results'].append(self._calculate_model_performance(model, dataloader_tr))
                cv_result['validation_results'].append(self._calculate_model_performance(model, dataloader_val))
            cv_results['training_results'].append(cv_result['training_results'])
            cv_results['validation_results'].append(cv_result['validation_results'])

        if not os.path.exists(os.path.join(self._results_directory, 'grid_search_scores', self._target_directory)):
            os.makedirs(os.path.join(self._results_directory, 'grid_search_scores', self._target_directory))

        with open(os.path.join(self._results_directory,
                               'grid_search_scores',
                               self._target_directory,
                               self._model_name + '_' + self._antibiotic_name + '.json'), 'w') as fp:
            json.dump(cv_results, fp)

        best_hyperparameters = choose_best_hyperparameters(cv_results, metric='f1')

        if not os.path.exists(os.path.join(self._results_directory, 'best_models', self._target_directory)):
            os.makedirs(os.path.join(self._results_directory, 'best_models', self._target_directory))

        with open(os.path.join(self._results_directory,
                               'best_models',
                               self._target_directory,
                               self._model_name + '_' + self._antibiotic_name + '.json'), 'w') as fp:
            json.dump(best_hyperparameters, fp)

    def predict_ar(self, x):
        self._model.eval()
        return self._model.forward(x)

    def test_model(self, x_te, y_te):
        self._best_model.to(self._device)

        self._best_model.eval()

        y_hat = self._best_model(torch.from_numpy(x_te).float().to(self._device))
        pred = torch.argmax(y_hat, dim=1)

        y_pred = pred.cpu().numpy()

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

    def train_best_model(self, hyperparameters, x_tr, y_tr, x_te, y_te):
        dataloader_tr = prepare_dataloader(hyperparameters['batch_size'], x_tr, y_tr)
        dataloader_te = prepare_dataloader(hyperparameters['batch_size'], x_te, y_te)

        # initialization of the model
        model = FeetForwardNetwork(self._feature_size,
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
        criterion = torch.nn.NLLLoss(reduction='mean')
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

        # create the directory holding checkpoints if not exists
        if not os.path.exists(os.path.join(self._results_directory, 'checkpoints')):
            os.makedirs(os.path.join(self._results_directory, 'checkpoints'))

        es = EarlyStopping(metric='loss',
                           mode='min',
                           patience=10,
                           checkpoint_file=os.path.join(self._results_directory,
                                                        'best_models',
                                                        self._target_directory,
                                                        self._model_name + '_' + self._antibiotic_name + '.pt'))

        for epoch in range(2000):
            model.train()

            training_results = None
            test_results = None

            # Training
            training_results = self._training_step(model, optimizer, criterion, dataloader_tr)

            # Validation
            test_results = self._validate_model(model, criterion, dataloader_te)

            print('[%d] training loss: %.9f' % (epoch, training_results['loss']))
            print('[%d] test loss: %.9f' % (epoch, test_results['loss']))

            if es.step(epoch, test_results, model):
                # print('Early stopping at epoch: ' + str(epoch) + ' best index: ' + str(es.best_index))
                print('Epoch: ' + str(es.best_index) + ', best metrics: ' + str(es.best_metrics))
                break
