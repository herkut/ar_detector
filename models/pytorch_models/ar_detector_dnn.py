import torch

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
        self.fcs = []
        self.bns = []

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
        self.predict = torch.nn.Linear(hidden_units[-1], self.output_size)

        for af in activation_functions:
            if af not in ['relu', 'tanh', 'hardtanh', 'leaky_relu']:
                # TODO define custom not implemented activation function exception
                raise Exception
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

        out = self.predict(x)
        return out


class PytorchARDetector(BaseARDetector):
    def load_model(self):
        pass

    def tune_hyperparameters(self, param_grid):
        pass

    def predict_ar(self, x):
        pass

    def test_model(self):
        pass

