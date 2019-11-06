import collections
import json
import os

import torch
from sklearn.metrics import confusion_matrix

from config import Config
from models.base_ar_detector import BaseARDetector
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


class ConvNet1D(torch.nn.Module):
    def __init__(self,
                 feature_size,
                 first_in_channel,
                 output_size,
                 conv_kernels,
                 conv_channels,
                 conv_strides,
                 conv_paddings,
                 conv_activation_functions,
                 pooling_kernels,
                 pooling_strides,
                 pooling_paddings,
                 fc_hidden_units,
                 fc_activation_functions,
                 fc_dropout_rate,
                 batch_normalization=False,
                 pooling_type=None):
        """

        :param feature_size:
        :param first_in_channel:
        :param output_size:
        :param conv_kernels:
        :param conv_channels:
        :param conv_strides:
        :param conv_activation_functions:
        :param pooling_kernels:
        :param pooling_strides:
        :param pooling_paddings:
        :param fc_hidden_units:
        :param fc_activation_functions:
        :param fc_dropout_rate:
        :param batch_normalization:
        :param pooling_type:
        """
        super(ConvNet1D, self).__init__()
        self.feature_size = feature_size
        self.first_in_channel = first_in_channel
        self.output_size = output_size

        self.do_bn = batch_normalization

        self.kernels = conv_kernels  # kernel_sizes
        self.channels = conv_channels  # channels
        self.strides = conv_strides  # strides
        self.conv_paddings = conv_paddings
        self.convs = []
        self.conv_bns = []
        self.conv_afs = conv_activation_functions

        self.pooling_type = pooling_type
        self.poolings = []
        self.pooling_kernels = pooling_kernels
        self.pooling_strides = pooling_strides
        self.pooling_paddings = pooling_paddings

        if fc_dropout_rate > 0:
            self.do_dropout = True
        else:
            self.do_dropout = False
        self.fc_hus = fc_hidden_units
        # fully connected layers
        self.fcs = []
        # batch normalizations
        self.fc_bns = []
        # dropouts
        self.fc_dos = []
        self.fc_afs = fc_activation_functions

        # TODO conduct hyperparameter arrays eg: len of conv kernels == len of conv afs etc
        # TODO same for fcs

        # Convolutional layers
        for i in range(len(self.kernels)):
            if not self.convs:
                if self.conv_paddings[i] is None:
                    self.convs.append(torch.nn.Conv1d(self.first_in_channel,
                                                      self.channels[i],
                                                      self.kernels[i],
                                                      stride=self.strides[i]))
                else:
                    self.convs.append(torch.nn.Conv1d(self.first_in_channel,
                                                      self.channels[i],
                                                      self.kernels[i],
                                                      stride=self.strides[i],
                                                      padding=conv_paddings[i]))
                setattr(self, 'conv%i' % i, self.convs[i])
            else:
                if self.conv_paddings[i] is None:
                    self.convs.append(torch.nn.Conv1d(self.channels[i-1],
                                                      self.channels[i],
                                                      self.kernels[i],
                                                      stride=self.strides[i]))
                else:
                    self.convs.append(torch.nn.Conv1d(self.channels[i-1],
                                                      self.channels[i],
                                                      self.kernels[i],
                                                      stride=self.strides[i],
                                                      padding=self.conv_paddings[i]))
                setattr(self, 'conv%i' % i, self.convs[i])

            if self.do_bn:
                self.conv_bns.append(torch.nn.BatchNorm1d(self.channels[i]))
                setattr(self, 'conv_bn%i' % i, self.conv_bns[i])

            if self.conv_afs[i] not in ['relu', 'tanh', 'hardtanh', 'leaky_relu']:
                # TODO define custom not implemented activation function exception
                raise Exception('Not implemented activation function: ' + self.conv_afs[i])
            # Pooling layer it is applied to just after all conv layers
            if self.pooling_type is not None and self.pooling_kernels[i] is not None and self.pooling_strides[i] is not None:
                if self.pooling_type == 'max':
                    if self.pooling_paddings[i] is None:
                        self.poolings.append(torch.nn.MaxPool1d(self.pooling_kernels[i],
                                                                stride=self.pooling_strides[i]))
                    else:
                        self.poolings.append(torch.nn.MaxPool1d(self.pooling_kernels[i],
                                                                stride=self.pooling_strides[i],
                                                                padding=self.pooling_paddings[i]))
                    setattr(self, 'pool%i' % i, self.poolings[i])
                elif self.pooling_type == 'average':
                    if self.pooling_paddings[i] is None:
                        self.poolings.append(torch.nn.AvgPool1d1d(self.pooling_kernels[i],
                                                                  stride=self.pooling_strides[i]))
                    else:
                        self.poolings.append(torch.nn.AvgPool1d1d(self.pooling_kernels[i],
                                                                  stride=self.pooling_strides[i],
                                                                  padding=self.pooling_paddings[i]))
                    setattr(self, 'pool%i' % i, self.poolings[i])
                #TODO raise unknown pooling method
            else:
                self.poolings.append(None)

        input_size_for_fcs = self._calculate_input_size_for_fc()

        # Fully connected layers
        for i in range(len(self.fc_hus)):
            # fc
            if not self.fcs:
                self.fcs.append(torch.nn.Linear(int(input_size_for_fcs), self.fc_hus[i]))
                setattr(self, 'fc%i' % i, self.fcs[i])
            else:
                self.fcs.append(torch.nn.Linear(self.fc_hus[i-1], self.fc_hus[i]))
                setattr(self, 'fc%i' % i, self.fcs[i])
            # bn
            if self.do_bn:
                self.fc_bns.append(torch.nn.BatchNorm1d(self.fc_hus[i]))
                setattr(self, 'fc_bn%i' % i, self.fc_bns[i])
            # do
            if self.do_dropout:
                self.fc_dos.append(torch.nn.Dropout(p=fc_dropout_rate))
                setattr(self, 'fc_do%i' % i, self.fc_dos[i])

            if self.fc_afs[i] not in ['relu', 'tanh', 'hardtanh', 'leaky_relu']:
                # TODO define custom not implemented activation function exception
                raise Exception('Not implemented activation function: ' + self.conv_afs[i])

        self.predict = torch.nn.Linear(self.fc_hus[-1], self.output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        # conv
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            if self.do_bn:
                x = self.conv_bns[i](x)

            # Set activations
            if self.conv_afs[i] == 'relu':
                x = torch.nn.ReLU()(x)
            elif self.conv_afs[i] == 'tanh':
                x = torch.nn.Tanh()(x)
            elif self.conv_afs[i] == 'hardtanh':
                x = torch.nn.Hardtanh()(x)
            elif self.conv_afs[i] == 'leaky_relu':
                x = torch.nn.LeakyReLU()(x)

            if self.poolings[i] is not None:
                x = self.poolings[i](x)

        # fc
        for i in range(len(self.fcs)):
            if i == 0:
                # like flatten in keras
                x = self.fcs[i](x.view(-1, x.shape[1]*x.shape[2]))
            else:
                x = self.fcs[i](x)
            if self.do_bn:
                x = self.fc_bns[i](x)
            if self.do_dropout:
                x = self.fc_dos[i](x)

            # Set activations
            if self.fc_afs[i] == 'relu':
                x = torch.nn.ReLU()(x)
            elif self.fc_afs[i] == 'tanh':
                x = torch.nn.Tanh()(x)
            elif self.fc_afs[i] == 'hardtanh':
                x = torch.nn.Hardtanh()(x)
            elif self.fc_afs[i] == 'leaky_relu':
                x = torch.nn.LeakyReLU()(x)

        out = self.predict(x)
        out = self.softmax(out + 1e-10)
        return out

    def _calculate_input_size_for_fc(self):
        # convolutional layers
        for i in range(len(self.kernels)):
            if i == 0:
                if self.conv_paddings[i] is None:
                    output_width = (self.feature_size - self.kernels[i] + 2 * 0) / self.strides[i] + 1
                else:
                    output_width = (self.feature_size - self.kernels[i] + 2 * self.conv_paddings[i]) / self.strides[i] + 1
            else:
                if self.conv_paddings[i] is None:
                    output_width = (output_width - self.kernels[i] + 2 * 0) / self.strides[i] + 1
                else:
                    output_width = (output_width - self.kernels[i] + 2 * self.conv_paddings[i]) / self.strides[i] + 1
            # pooling
            if self.poolings[i] is not None:
                output_width = (output_width - self.poolings[i].kernel_size + 2 * self.poolings[i].padding) / self.poolings[i].stride + 1

        return output_width * self.channels[-1]


class ARDetectorCNN(BaseARDetector):
    # TODO convert multilabel classifier and use only samples which have label for all antibiotics
    def __init__(self, feature_size, first_in_channel, output_size, antibiotic_name=None, model_name='cnn', class_weights=None):
        self._results_directory = Config.results_directory
        self._dataset = Config.cnn_target_dataset
        self._results_directory = self._results_directory + '_' + self._dataset

        # self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # TODO do below
            # instead of setting visible devices to least used cuda device
            # set visible devices with all cuda devices and use the least used one
            cuda_env_var, least_used_cuda = get_least_used_cuda_device()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(least_used_cuda)
            self._device = torch.device("cuda:0")
        else:
            self._device = "cpu"

        self._feature_size = feature_size
        self._first_in_channel = first_in_channel
        self._output_size = output_size
        self._antibiotic_name = antibiotic_name
        self._model_name = model_name
        if class_weights is not None:
            self._class_weights = class_weights[:, 1]
        else:
            self._class_weights = None
        self._scoring = Config.deep_learning_metric
        self._label_tags = Config.label_tags

        self._target_directory = self._model_name + '_' + self._scoring + '_' + self._label_tags

    def _initialize_model(self, device, feature_size, first_in_channel, output_size, class_weights, hyperparameters):
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
        model.to(device)

        if class_weights is not None:
            criterion = torch.nn.NLLLoss(reduction='mean', weight=torch.from_numpy(class_weights).to(device))
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

        model, _, _ = self._initialize_model(self._device,
                                             self._feature_size,
                                             self._first_in_channel,
                                             self._output_size,
                                             self._class_weights,
                                             best_hyperparameters)
        model.load_state_dict(torch.load(os.path.join(self._results_directory,
                                                      'best_models',
                                                      self._target_directory,
                                                      self._model_name + '_' + self._antibiotic_name + '.pt')))
        model.to(self._device)
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
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)

            # initialization of gradients
            optimizer.zero_grad()
            # Forward propagation
            y_hat = model(inputs.float())
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

    def _validate_model(self, model, criterion, dataloader):
        model.eval()
        val_loss = 0
        validation_results = None
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, labels = data
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)

                # Forward propagation
                y_hat = model(inputs.float())
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
        hyperparameter_space = create_hyperparameter_space_for_cnn(param_grid)

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
                model, criterion, optimizer = self._initialize_model(self._device,
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
                                   required_min_iteration=15)

                self._train_model(model, criterion, optimizer, es, tr_dataloader, val_dataloader)
                # Training has been completed
                # Validate model with best weights after early stopping
                model, criterion, _ = self._initialize_model(self._device,
                                                             self._feature_size,
                                                             self._first_in_channel,
                                                             self._output_size,
                                                             self._class_weights,
                                                             grid)
                model.load_state_dict(torch.load(os.path.join(self._results_directory,
                                                              'checkpoints',
                                                              self._model_name + '_checkpoint.pt')))
                model.to(self._device)
                # best model performance on training data for these data folds and hyperparameters
                cv_result['training_results'].append(self._validate_model(model, criterion, tr_dataloader))
                # best model performance on validation data for these data folds and hyperparameters
                cv_result['validation_results'].append(self._validate_model(model, criterion, val_dataloader))

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

    def train_best_model(self, hyperparameters, idx_tr, labels_tr, idx_te, labels_te):
        bs = hyperparameters['batch_size']

        dataset_tr = ARCNNDataset(idx_tr, labels_tr, self._antibiotic_name)
        dataloader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=bs)

        dataset_te = ARCNNDataset(idx_te, labels_te, self._antibiotic_name)
        dataloader_te = torch.utils.data.DataLoader(dataset_te, batch_size=bs)

        model, criterion, optimizer = self._initialize_model(self._device,
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
                           required_min_iteration=15)

        self._train_model(model, criterion, optimizer, es, dataloader_tr, dataloader_te)

    def predict_ar(self, x):
        pass

    def test_model(self, idx, labels):
        self._best_model.to(self._device)

        self._best_model.eval()

        dataset = ARCNNDataset(idx, labels, self._antibiotic_name)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

        pred = None
        actual_labels = None
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, batch_labels = data
                inputs = inputs.to(self._device)

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
