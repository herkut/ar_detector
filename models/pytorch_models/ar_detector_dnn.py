import json
import os

import torch
import torch.utils.data as utils

from config import Config
from models.base_ar_detector import BaseARDetector
from utils.confusion_matrix_drawer import classification_report, concatenate_classification_reports
from utils.helper_functions import get_k_fold_validation_indices, create_hyperparameter_space
from utils.statistical_tests.statistical_tests import choose_best_hyperparameters, compare_hyperparameters_wrt_corrected_1xkcv_t_test


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
            self.save_checkpoint(epoch, results, model)
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
                ar_dataloader_tr = prepare_dataloader(batch_size, x_tr[train_indeces], y_tr[train_indeces])
                ar_dataloader_val = prepare_dataloader(batch_size, x_tr[validation_indeces], y_tr[validation_indeces])

                feature_size = ar_dataloader_tr.dataset.tensors[0].shape[1]

                self._train_for_cv(grid, ar_dataloader_tr, ar_dataloader_val)

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

        if not os.path.exists(os.path.join(Config.results_directory, 'grid_search_scores', self._target_directory)):
            os.makedirs(os.path.join(Config.results_directory, 'grid_search_scores', self._target_directory))

        with open(os.path.join(Config.results_directory,
                               'grid_search_scores',
                               self._target_directory,
                               self._model_name + '_' + self._antibiotic_name + '.json'), 'w') as fp:
            json.dump(cv_results, fp)

        # find best hyperparameters train on whole train set and save the model in the file
        best_hyperparameters = self._choose_best_hyperparameters(cv_results)

        if not os.path.exists(os.path.join(Config.results_directory, 'best_models', self._target_directory)):
            os.makedirs(os.path.join(Config.results_directory, 'best_models', self._target_directory))

        with open(os.path.join(Config.results_directory,
                               'best_models',
                               self._target_directory,
                               'best_hyperparameters_' + self._model_name + '_' + self._antibiotic_name + '.json'), 'w') as fp:
            json.dump(best_hyperparameters, fp)

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
        pred = None
        labels = None
        for i, data in enumerate(dataloader, 0):
            inputs, labels_tmp = data
            inputs = inputs.to(self._device)
            labels_tmp = labels_tmp.to(self._device)

            if labels is None:
                labels = labels_tmp
            else:
                labels = torch.cat((labels, labels_tmp), 0)

            y_hat = model(inputs)
            if pred is None:
                pred = torch.argmax(y_hat, dim=1)
            else:
                pred = torch.cat((pred, torch.argmax(y_hat, dim=1)), 0)

            cost = criterion(y_hat, labels_tmp)
            validation_loss += cost
            """
            tmp_classification_report = classification_report(labels_tmp, pred)
            if validation_results is None:
                validation_results = tmp_classification_report
            else:
                validation_results = concatenate_classification_reports(validation_results,
                                                                        tmp_classification_report)
            """
        validation_results = classification_report(labels, pred)
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

    def _choose_best_hyperparameters(self, cv_results, metric='f1'):
        # TODO make hyperparameters comparing methods generic (rxkcv)
        best_hyperparameter_id = 0
        for i in range(1, len(cv_results['grids'])):
            print('Comparing: ')
            print(cv_results['grids'][best_hyperparameter_id])
            print(cv_results['grids'][i])
            _, res = compare_hyperparameters_wrt_corrected_1xkcv_t_test(cv_results['validation_results'][best_hyperparameter_id],
                                                                        cv_results['validation_results'][i],
                                                                        metric=metric)
            if res == -1:
                best_hyperparameter_id = i

        print('Best hyperparameter index: ' + str(best_hyperparameter_id))
        print(cv_results['grids'][best_hyperparameter_id])

        return cv_results['grids'][best_hyperparameter_id]

    def _train_for_cv(self, hyperparameters, ar_dataloader_tr, ar_dataloader_val):
        feature_size = ar_dataloader_tr.dataset.tensors[0].shape[1]

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

        # create the directory holding checkpoints if not exists
        if not os.path.exists(os.path.join(Config.results_directory, 'checkpoints')):
            os.makedirs(os.path.join(Config.results_directory, 'checkpoints'))

        es = EarlyStopping(metric='f1',
                           mode='max',
                           patience=10,
                           checkpoint_file=os.path.join(Config.results_directory, 'checkpoints',
                                                        self._model_name + '_checkpoint.pt'))

        for epoch in range(2000):
            model.train()

            # Training
            training_results = self._training_step(model, optimizer, criterion, ar_dataloader_tr)

            # Validation
            validation_results = self._validate_model(model, criterion, ar_dataloader_val)

            if es.step(epoch, validation_results, model):
                # print('Early stopping at epoch: ' + str(epoch) + ' best index: ' + str(es.best_index))
                print('Best epoch: ' + str(es.best_index) + ', best metrics: ' + str(es.best_metrics))
                break

            print('[%d] training loss: %.9f, f1: %.9f' % (epoch + 1, training_results['loss'], training_results['f1']))
            print('[%d] validation loss: %.9f, f1: %.9f' % (epoch + 1, validation_results['loss'], validation_results['f1']))

    def train_and_test_model(self, x_tr, y_tr, x_te, y_te):
        pass