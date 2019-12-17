import collections
import json
import os

import torch
from sklearn.metrics import confusion_matrix
from pynvml import *

from config import Config
from models.base_ar_detector import BaseARDetector
from models.pytorch_models.conv_nets import ConvNet1D, ConvNet0
from models.pytorch_models.early_stopping import EarlyStopping
from preprocess.cnn_dataset import ARCNNDataset
from preprocess.feature_label_preparer import FeatureLabelPreparer
from utils.confusion_matrix_drawer import classification_report, concatenate_classification_reports, \
    plot_confusion_matrix
from utils.helper_functions import get_index_to_remove, get_k_fold, create_hyperparameter_space_for_cnn, \
    get_least_used_cuda_device
import numpy as np
import matplotlib.pyplot as plt
from utils.statistical_tests.statistical_tests import choose_best_hyperparameters
import pandas as pd


class ARDetectorCNN(BaseARDetector):
    # TODO convert multilabel classifier and use only samples which have label for all antibiotics
    def __init__(self,
                 feature_size,
                 first_in_channel,
                 output_size,
                 antibiotic_name=None,
                 model_name='cnn',
                 class_weights=None,
                 gpu_count=1):
        self._results_directory = Config.results_directory
        self._dataset = Config.cnn_target_dataset
        self._results_directory = self._results_directory + '_' + self._dataset
        self._model_name = model_name

        # self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # TODO do below
            # instead of setting visible devices to least used cuda device
            # set visible devices with all cuda devices and use the least used one
            if self._model_name == 'conv_0':
                cuda_env_var, least_used_gpus = get_least_used_cuda_device(gpu_count=gpu_count)
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_env_var
            else:
                cuda_env_var, least_used_gpus = get_least_used_cuda_device(gpu_count=1)
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_env_var
                # if there is only one valid gpu in cuda visible devices pytorch see gpu
                # as 0 without minding its original id
                least_used_gpus = [0]

            self._devices = least_used_gpus[::-1]
        else:
            self._devices = "cpu"

        self._feature_size = feature_size
        self._first_in_channel = first_in_channel
        self._output_size = output_size
        self._antibiotic_name = antibiotic_name
        if class_weights is not None:
            self._class_weights = class_weights[:, 1]
        else:
            self._class_weights = None
        self._scoring = Config.deep_learning_metric
        self._label_tags = Config.label_tags

        self._target_directory = self._model_name + '_' + self._scoring + '_' + self._label_tags

    def _initialize_model(self, devices, feature_size, first_in_channel, output_size, class_weights, hyperparameters):
        """
        feature_size,
        output_size,
        conv_kernels,
        conv_channels,
        conv_strides,
        conv_activation_functions,
        pooling_kernels,
        pooling_strides,
        fc_hidden_units,
        fc_activation_functions,
        fc_dropout_rate,
        batch_normalization=False,
        pooling_type=None
        """
        if self._model_name == 'conv_0':
            model = ConvNet0(self._devices,
                             feature_size,
                             first_in_channel,
                             output_size,
                             hyperparameters['conv_kernels'],
                             hyperparameters['conv_channels'],
                             hyperparameters['conv_strides'],
                             hyperparameters['conv_paddings'],
                             hyperparameters['conv_activation_functions'],
                             hyperparameters['fc_hidden_units'],
                             hyperparameters['fc_activation_functions'],
                             hyperparameters['fc_dropout_rate'],
                             batch_normalization=True,
                             pooling_type=hyperparameters['pooling_type'])
            if class_weights is not None:
                criterion = torch.nn.NLLLoss(reduction='mean',
                                             weight=torch.from_numpy(class_weights).to('cuda:'+str(devices[3])))
            else:
                criterion = torch.nn.NLLLoss(reduction='mean')
        else:
            model = ConvNet1D(feature_size,
                              first_in_channel,
                              output_size,
                              hyperparameters['conv_kernels'],
                              hyperparameters['conv_channels'],
                              hyperparameters['conv_strides'],
                              hyperparameters['conv_paddings'],
                              hyperparameters['conv_activation_functions'],
                              hyperparameters['pooling_kernels'],
                              hyperparameters['pooling_strides'],
                              hyperparameters['pooling_paddings'],
                              hyperparameters['fc_hidden_units'],
                              hyperparameters['fc_activation_functions'],
                              hyperparameters['fc_dropout_rate'],
                              batch_normalization=True,
                              pooling_type=hyperparameters['pooling_type'])
            # TODO distribute model over multiple gpus for proposed architectures
            model.to('cuda:' + str(devices[0]))

            if class_weights is not None:
                criterion = torch.nn.NLLLoss(reduction='mean',
                                             weight=torch.from_numpy(class_weights).to('cuda:' + str(devices[0])))
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

        return model, criterion, optimizer

    def load_model(self):
        with open(os.path.join(self._results_directory,
                               'best_models',
                               self._target_directory,
                               self._model_name + '_' + self._antibiotic_name + '.json')) as fp:
            best_hyperparameters = json.load(fp)

        model, _, _ = self._initialize_model(self._devices,
                                             self._feature_size,
                                             self._first_in_channel,
                                             self._output_size,
                                             self._class_weights,
                                             best_hyperparameters)
        model.load_state_dict(torch.load(os.path.join(self._results_directory,
                                                      'best_models',
                                                      self._target_directory,
                                                      self._model_name + '_' + self._antibiotic_name + '.pt')))
        model.eval()

        self._best_model = model

    def save_model(self):
        pass

    def _training_step(self, model, criterion, optimizer, dataloader):
        model.train()
        tr_loss = 0
        training_results = None
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to('cuda:' + str(self._devices[-1]))
            labels = labels.to('cuda:' + str(self._devices[-1]))

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

            # Reporting
            tr_loss += cost
            tmp_classification_report = classification_report(labels, pred)
            if training_results is None:
                training_results = tmp_classification_report
            else:
                training_results = concatenate_classification_reports(training_results,
                                                                      tmp_classification_report)
        training_results['loss'] = tr_loss
        return training_results

    def _validate_model(self, model, criterion, dataloader, ignore_loss=False):
        model.eval()
        val_loss = 0
        validation_results = None
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, labels = data
                inputs = inputs.to('cuda:' + str(self._devices[-1]))
                labels = labels.to('cuda:' + str(self._devices[-1]))

                # Forward propagation
                y_hat = model(inputs)
                pred = torch.argmax(y_hat, dim=1)
                # Computation of cost function
                cost = criterion(y_hat, labels)

                # Reporting
                val_loss += cost
                tmp_classification_report = classification_report(labels, pred)
                if validation_results is None:
                    validation_results = tmp_classification_report
                else:
                    validation_results = concatenate_classification_reports(validation_results,
                                                                            tmp_classification_report)
        if not ignore_loss:
            validation_results['loss'] = val_loss
        return validation_results

    def _train_model(self, model, criterion, optimizer, es, tr_dataloader, val_dataloader):
        for epoch in range(200):
            # Training
            tr_results = self._training_step(model, criterion, optimizer, tr_dataloader)
            print(str(epoch) + ': tr loss ' + str(tr_results['loss']))

            # Validation
            val_results = self._validate_model(model, criterion, val_dataloader)
            print(str(epoch) + ': val loss ' + str(val_results['loss']))

            if es.step(epoch, val_results, model):
                # print('Early stopping at epoch: ' + str(epoch) + ' best index: ' + str(es.best_index))
                print('Epoch: ' + str(es.best_index) + ', best metrics: ' + str(es.best_metrics))
                break

    def tune_hyperparameters(self, param_grid, idx, labels):
        hyperparameter_space = create_hyperparameter_space_for_cnn(param_grid, self._model_name)

        cv_results = {'grids': [], 'training_results': [], 'validation_results': []}
        for grid in hyperparameter_space:
            print('Grid: ' + str(grid))
            bs = grid['batch_size']
            cv_results['grids'].append(grid)
            cv_result = {'training_results': [], 'validation_results': []}

            cv = get_k_fold(10)

            for tr_index, val_index in cv.split(idx, labels):
                # Training dataset
                tr_dataset = ARCNNDataset(idx[tr_index], labels[tr_index], self._antibiotic_name)
                tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=bs)

                # Validation dataset
                val_dataset = ARCNNDataset(idx[val_index], labels[val_index], self._antibiotic_name)
                val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=bs)

                # initialize the convolutional model
                model, criterion, optimizer = self._initialize_model(self._devices,
                                                                     self._feature_size,
                                                                     self._first_in_channel,
                                                                     self._output_size,
                                                                     self._class_weights,
                                                                     grid)

                # create the directory holding checkpoints if not exists
                if not os.path.exists(os.path.join(self._results_directory, 'checkpoints')):
                    os.makedirs(os.path.join(self._results_directory, 'checkpoints'))

                es = EarlyStopping(metric='loss',
                                   mode='min',
                                   patience=10,
                                   checkpoint_file=os.path.join(self._results_directory,
                                                                'checkpoints',
                                                                self._model_name + '_checkpoint.pt'),
                                   required_min_iteration=20)

                self._train_model(model, criterion, optimizer, es, tr_dataloader, val_dataloader)
                # Training has been completed
                # Validate model with best weights after early stopping
                model, criterion, _ = self._initialize_model(self._devices,
                                                             self._feature_size,
                                                             self._first_in_channel,
                                                             self._output_size,
                                                             self._class_weights,
                                                             grid)
                model.load_state_dict(torch.load(os.path.join(self._results_directory,
                                                              'checkpoints',
                                                              self._model_name + '_checkpoint.pt')))

                # best model performance on training data for these data folds and hyperparameters
                cv_result['training_results'].append(self._validate_model(model, criterion, tr_dataloader, ignore_loss=True))
                # best model performance on validation data for these data folds and hyperparameters
                cv_result['validation_results'].append(self._validate_model(model, criterion, val_dataloader, ignore_loss=True))

            cv_results['training_results'].append(cv_result['training_results'])
            cv_results['validation_results'].append(cv_result['validation_results'])

        if not os.path.exists(os.path.join(self._results_directory, 'grid_search_scores', self._target_directory)):
            os.makedirs(os.path.join(self._results_directory, 'grid_search_scores', self._target_directory))

        with open(os.path.join(self._results_directory,
                               'grid_search_scores',
                               self._target_directory,
                               self._model_name + '_' + self._antibiotic_name + '.json'), 'w') as fp:
            json.dump(cv_results, fp)

        best_hyperparameters = choose_best_hyperparameters(cv_results, metric=self._scoring)

        if not os.path.exists(os.path.join(self._results_directory, 'best_models', self._target_directory)):
            os.makedirs(os.path.join(self._results_directory, 'best_models', self._target_directory))

        with open(os.path.join(self._results_directory,
                               'best_models',
                               self._target_directory,
                               self._model_name + '_' + self._antibiotic_name + '.json'), 'w') as fp:
            json.dump(best_hyperparameters, fp)

    def train_best_model(self, hyperparameters, idx_tr, labels_tr, idx_te, labels_te):
        bs = hyperparameters['batch_size']

        dataset_tr = ARCNNDataset(idx_tr, labels_tr, self._antibiotic_name)
        dataloader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=bs)

        dataset_te = ARCNNDataset(idx_te, labels_te, self._antibiotic_name)
        dataloader_te = torch.utils.data.DataLoader(dataset_te, batch_size=bs)

        model, criterion, optimizer = self._initialize_model(self._devices,
                                                             self._feature_size,
                                                             self._first_in_channel,
                                                             self._output_size,
                                                             self._class_weights,
                                                             hyperparameters)

        # create the directory holding checkpoints if not exists
        if not os.path.exists(os.path.join(self._results_directory, 'checkpoints')):
            os.makedirs(os.path.join(self._results_directory, 'checkpoints'))

        es = EarlyStopping(metric='loss',
                           mode='min',
                           patience=10,
                           checkpoint_file=os.path.join(self._results_directory,
                                                        'best_models',
                                                        self._target_directory,
                                                        self._model_name + '_' + self._antibiotic_name + '.pt'),
                           required_min_iteration=20)

        self._train_model(model, criterion, optimizer, es, dataloader_tr, dataloader_te)

    def predict_ar(self, x):
        pass

    def test_model(self, idx, labels):
        self._best_model.eval()

        dataset = ARCNNDataset(idx, labels, self._antibiotic_name)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

        pred = None
        actual_labels = None
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, batch_labels = data

                if actual_labels is None:
                    actual_labels = batch_labels
                else:
                    torch.cat((actual_labels, batch_labels), 0)

                y_hat = self._best_model(inputs)
                if pred is None:
                    pred = torch.argmax(y_hat, dim=1)
                else:
                    pred = torch.cat((pred, torch.argmax(y_hat, dim=1)), 0)

        y_pred = pred.cpu().numpy()

        cm = confusion_matrix(labels, y_pred)
        if np.shape(cm)[0] == 2 and np.shape(cm)[1] == 2:
            sensitivity = float(cm[0][0]) / np.sum(cm[0])
            specificity = float(cm[1][1]) / np.sum(cm[1])
            print('For ' + self._antibiotic_name)
            print(collections.Counter(labels))
            print('Sensitivity: ' + str(sensitivity))
            print('Specificity: ' + str(specificity))
        else:
            print('For ' + self._antibiotic_name)
            print('There has been an error in calculating sensitivity and specificity')

        plot_confusion_matrix(labels,
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

        plot_confusion_matrix(labels,
                              y_pred,
                              classes=['susceptible', 'resistant'],
                              normalize=False,
                              title='Confusion matrix')

        plt.savefig(os.path.join(self._results_directory,
                                 'confusion_matrices',
                                 self._target_directory,
                                 self._model_name + '_' + self._antibiotic_name + '.png'))

        y_true = pd.Series(labels, name="Actual")
        y_pred = pd.Series(y_pred, name="Predicted")
        df_confusion = pd.crosstab(y_true, y_pred)
        df_confusion.to_csv(os.path.join(self._results_directory,
                                         'confusion_matrices',
                                         self._target_directory,
                                         self._model_name + '_' + self._antibiotic_name + '.csv'))
