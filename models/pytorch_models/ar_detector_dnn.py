import torch
import torch.utils.data as utils

from config import Config
from models.base_ar_detector import BaseARDetector


class FeetForwardNetwork(torch.nn.module):
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
                bn = torch.nn.BatchNorm1d(hidden_units[0], momentum=0.5)
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
            if self.do_bn:
                x = self.bns[i](x)
            # Set activations
            if self.activations[i] == 'relu':
                x = torch.nn.ReLU(x)
            elif self.activations[i] == 'tanh':
                x = torch.nn.Tanh(x)
            elif self.activations[i] == 'hardtanh':
                x = torch.nn.Hardtanh(x)
            elif self.activations[i] == 'leaky_relu':
                x = torch.nn.LeakyReLU(x)

            if self.do_dropout:
                x = self.dos[i](x)

        out = self.predict(x)
        return out

    def get_name(self):
        return 'dnn-' + str(len(self.fcs)) + 'd'


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
        self._class_weights = class_weights

    def save_model(self):
        pass

    def load_model(self):
        pass

    def tune_hyperparameters(self, param_grid, x_tr, y_tr):
        # put model in gpu if exists
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        # x_tr[0] sample counts x_tr[1] feature size
        feature_size = x_tr.shape[1]

        # create dataset and dataloader
        # convert labels into one hot encoded
        y_tr_one_hot = torch.nn.functional.one_hot(y_tr, num_classes=2)
        x_tr_tensor = torch.stack([torch.Tensor(i) for i in x_tr])  # transform to torch tensors
        ar_dataset = utils.TensorDataset(x_tr_tensor, y_tr_one_hot)
        ar_dataloader = utils.DataLoader(ar_dataset, batch_size=64)

        # grid search
        for optimizer_param in param_grid['optimizers']:
            for lr in param_grid['learning_rates']:
                for hu in param_grid['hidden_units']:
                    for af in param_grid['activation_functions']:
                        for dr in param_grid['dropout_rates']:
                            # initialization of the model
                            self._model = FeetForwardNetwork(feature_size, 2, hu, af, dr, batch_normalization=True)
                            self._model_name = self._model.get_name()
                            self._target_directory = self._model_name + '_' + self._scoring + '_' + self._label_tags + '_' + self._feature_selection
                            # initialization completed

                            # Optimizer initialization
                            criterion = torch.nn.BCEWithLogitsLoss()  # BCELoss with sigmoid in front of it
                            if optimizer_param == 'Adam':
                                optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
                            elif optimizer_param == 'SGD':
                                optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
                            elif optimizer_param == 'Adamax':
                                optimizer = torch.optim.Adamax(self._model.parameters(), lr=lr)
                            elif optimizer_param == 'RMSProp':
                                optimizer = torch.optim.RMSProp(self._model.parameters(), lr=lr)
                            else:
                                raise Exception('Not implemented optimizer: ' + optimizer_param)
                            # Optimizer initialization completed

                            # Train the model
                            # TODO implement early stopping
                            for epoch in range(2000):
                                running_loss = 0.0
                                for i, data in enumerate(ar_dataloader, 0):
                                    inputs, labels = data
                                    # initialization of gradients
                                    optimizer.zero_grad()
                                    # Forward propagation
                                    y_pred = self._model(inputs)
                                    # Computation of cost function
                                    cost = criterion(y_pred, labels, pos_weight=torch.from_numpy(self._class_weights))
                                    # Back propagation
                                    cost.backward()
                                    # Update parameters
                                    optimizer.step()

                                    running_loss += cost

                                    if i % 40 == 0:  # print every 2000 mini-batches
                                        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                                        running_loss = 0.0
                            # End of training

    def predict_ar(self, x):
        self._model.eval()
        return self._model.forward(x)

    def test_model(self, x_te, y_te):
        self._model.eval()
        pass
