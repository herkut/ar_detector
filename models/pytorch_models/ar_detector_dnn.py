import torch
import torch.utils.data as utils

import numpy as np

from config import Config
from models.base_ar_detector import BaseARDetector
from utils.helper_functions import get_k_fold_validation_indices, create_hyperparameter_space


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
        return out

    def get_name(self):
        return 'dnn-' + str(len(self.fcs)) + 'd'


def prepate_dataloaders(batch_size, x_tr, y_tr, train_indices, validation_indices):
    x_tr_tensor = torch.from_numpy(x_tr[train_indices]).float()  # numpy matrix -> torch tensors
    y_tr_tensor = torch.from_numpy(y_tr[train_indices])  # numpy array -> numpy matrix (len, 1) -> torch tensors

    x_val_tensor = torch.from_numpy(x_tr[validation_indices]).float()  # numpy matrix -> torch tensors
    y_val_tensor = torch.from_numpy(y_tr[validation_indices])  # numpy array -> numpy matrix (len, 1) -> torch tensors

    # convert labels into one hot encoded
    y_tr_one_hot = torch.nn.functional.one_hot(y_tr_tensor.to(torch.int64), num_classes=2).float()
    y_val_one_hot = torch.nn.functional.one_hot(y_val_tensor.to(torch.int64), num_classes=2).float()

    # create dataset and dataloader
    # convert elements of tensor to int64 for label matrix,
    # otherwise pytorch would throw an error 'RuntimeError: one_hot is only applicable to index tensor.'
    ar_dataset_tr = utils.TensorDataset(x_tr_tensor, y_tr_one_hot)
    ar_dataloader_tr = utils.DataLoader(ar_dataset_tr, batch_size=batch_size)
    ar_dataset_val = utils.TensorDataset(x_val_tensor, y_val_one_hot)
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

        self._target_directory = None
        self._model_name = model_name
        self._class_weights = class_weights[:, 1]

    def save_model(self):
        pass

    def load_model(self):
        pass

    def tune_hyperparameters(self, param_grid, x_tr, y_tr):
        batch_size = 64
        # put model in gpu if exists
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x_tr[0] sample counts x_tr[1] feature size
        feature_size = x_tr.shape[1]

        # prepare data for closs validation
        k_fold_indices = get_k_fold_validation_indices(10, x_tr, y_tr)

        # grid search
        hyperparameter_space = create_hyperparameter_space(param_grid)

        for grid in hyperparameter_space:
            for train_indeces, validation_indeces in k_fold_indices:
                # initialization of dataloaders
                ar_dataloader_tr, ar_dataloader_val = prepate_dataloaders(batch_size, x_tr, y_tr, train_indeces, validation_indeces)

                # initialization of the model
                self._model = FeetForwardNetwork(feature_size,
                                                 2,
                                                 grid['hidden_units'],
                                                 grid['activation_functions'],
                                                 grid['dropout_rate'],
                                                 batch_normalization=True)
                # put model into gpu if exists
                self._model.to(device)
                self._model_name = self._model.get_name()
                self._target_directory = self._model_name + '_' + self._scoring + '_' + self._label_tags + '_' + self._feature_selection
                # initialization completed

                # Optimizer initialization
                # class weight is used to handle unbalanced data
                # BCEWithLogitsLoss = BCELoss with sigmoid in front of it
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(self._class_weights))
                if grid['optimizer'] == 'Adam':
                    optimizer = torch.optim.Adam(self._model.parameters(), lr=grid['learning_rate'])
                elif grid['optimizer'] == 'SGD':
                    optimizer = torch.optim.SGD(self._model.parameters(), lr=grid['learning_rate'])
                elif grid['optimizer'] == 'Adamax':
                    optimizer = torch.optim.Adamax(self._model.parameters(), lr=grid['learning_rate'])
                elif grid['optimizer'] == 'RMSProp':
                    optimizer = torch.optim.RMSProp(self._model.parameters(), lr=grid['learning_rate'])
                else:
                    raise Exception('Not implemented optimizer: ' + grid['optimizer'])
                # Optimizer initialization completed

                # TODO implement early stopping
                # TODO create result matrix containing tp, fp, tn, fn, accuracy, f1_score etc.
                for epoch in range(2000):
                    self._model.train()
                    training_loss = 0.0
                    # Training
                    for i, data in enumerate(ar_dataloader_tr, 0):
                        inputs, labels = data
                        # initialization of gradients
                        optimizer.zero_grad()
                        # Forward propagation
                        y_pred = self._model(inputs)
                        # Computation of cost function
                        cost = criterion(y_pred, labels)
                        # Back propagation
                        cost.backward()
                        # Update parameters
                        optimizer.step()

                        training_loss += cost
                    # Validation
                    validation_loss = 0.0
                    self._model.eval()
                    for i, data in enumerate(ar_dataloader_val, 0):
                        y_pred = self._model(inputs)
                        cost = criterion(y_pred, labels)
                        validation_loss += cost

                    print('[%d, %5d] training loss: %.9f' % (epoch + 1, i + 1, training_loss / 2000))
                    print('[%d, %5d] validation loss: %.9f' % (epoch + 1, i + 1, validation_loss / 2000))

    def predict_ar(self, x):
        self._model.eval()
        return self._model.forward(x)

    def test_model(self, x_te, y_te):
        self._model.eval()
        pass
