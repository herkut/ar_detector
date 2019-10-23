import torch

from models.base_ar_detector import BaseARDetector


class ConvNet1D(torch.nn.Module):
    def __init__(self,
                 feature_size,
                 first_in_channel,
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
                 pooling_type=None):
        super(ConvNet1D, self).__init__()
        self.feature_size = feature_size
        self.first_in_channel = first_in_channel
        self.output_size = output_size

        self.do_bn = batch_normalization

        self.kernels = conv_kernels  # kernel_sizes
        self.channels = conv_channels  # channels
        self.strides = conv_strides  # strides
        self.convs = []
        self.conv_bns = []
        self.conv_afs = conv_activation_functions

        self.pooling_type = pooling_type
        self.poolings = []
        self.pooling_kernels = pooling_kernels
        self.pooling_strides = pooling_strides

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
                self.convs.append(torch.nn.Conv1d(self.first_in_channel,
                                                  self.channels[i],
                                                  self.kernels[i],
                                                  stride=self.strides[i]))
                setattr(self, 'conv%i' % i, self.convs[i])
            else:
                self.convs.append(torch.nn.Conv1d(self.channels[i-1],
                                                  self.channels[i],
                                                  self.kernels[i],
                                                  stride=self.strides[i]))
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
                    self.poolings.append(torch.nn.MaxPool1d(self.pooling_kernels[i], stride=self.pooling_strides[i]))
                    setattr(self, 'pool%i' % i, self.poolings[i])
                elif self.pooling_type == 'average':
                    self.pooling.append(torch.nn.AvgPool1d1d(self.pooling_kernels[i], stride=self.pooling_strides[i]))
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

        # pooling
        x = self.pooling(x)

        # fc
        for i in range(len(self.fcs)):
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
                output_width = (self.feature_size - self.kernels[i] + 2 * 0) / self.strides[i] + 1
            else:
                output_width = (output_width - self.kernels[i] + 2 * 0) / self.strides[i] + 1
            # pooling
            if self.poolings[i] is not None:
                output_width = (output_width - self.poolings[i].kernel_size + 2 * self.poolings[i].padding) / self.poolings[i].stride + 1

        return output_width * self.channels[-1]


class ARDetectorCNN(BaseARDetector):
    def __init__(self):
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass

    def set_class_weights(self, class_weights):
        pass

    def tune_hyperparameters(self, param_grid, x_tr, y_tr):
        pass

    def train_best_model(self, hyperparameters, x_tr, y_tr, x_te, y_te):
        pass

    def predict_ar(self, x):
        pass

    def test_model(self, x_te, y_te):
        pass


if __name__ == '__main__':
    """
    feature_size,
    output_size,
    conv_kernels,
    conv_channels,
    conv_strides,
    conv_activation_functions,
    fc_hidden_units,
    fc_activation_functions,
    fc_dropout_rate,
    batch_normalization=False,
    pooling_type=None
    """

    feature_size = 37733
    first_in_channel = 5
    output_size = 2

    conv_net = ConvNet1D(feature_size,
                         first_in_channel,
                         output_size,
                         [16, 32],
                         [64, 128],
                         [1, 1],
                         ['relu', 'relu'],
                         [None, 13],
                         [None, 13],
                         [1024, 256],
                         ['relu', 'relu'],
                         0,
                         batch_normalization=True,
                         pooling_type='max')

    print(conv_net)