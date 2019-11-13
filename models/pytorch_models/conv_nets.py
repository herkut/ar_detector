import torch
import numpy as np


class ConvNet0(torch.nn.Module):
    """
    Be inspired by the paper named "Convolutional Neural Networks for Sentence Classification" written by Yoon Kim
    """
    def __init__(self,
                 devices,
                 feature_size,
                 first_in_channel,
                 output_size,
                 conv_kernels,
                 conv_channels,
                 conv_strides,
                 conv_paddings,
                 conv_activation_functions,
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
        :param fc_hidden_units:
        :param fc_activation_functions:
        :param fc_dropout_rate:
        :param batch_normalization:
        :param pooling_type:
        """
        super(ConvNet0, self).__init__()
        self.devices = devices
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

        # Convolutional layers
        for i in range(len(self.kernels)):
            if i < 2:
                if self.conv_paddings[i] is None:
                    self.convs.append(torch.nn.Conv1d(self.first_in_channel,
                                                      self.channels[i],
                                                      self.kernels[i],
                                                      stride=self.strides[i]).to('cuda:'+str(devices[0])))
                else:
                    self.convs.append(torch.nn.Conv1d(self.first_in_channel,
                                                      self.channels[i],
                                                      self.kernels[i],
                                                      stride=self.strides[i],
                                                      padding=conv_paddings[i]).to('cuda:'+str(devices[0])))
                setattr(self, 'conv%i' % i, self.convs[i])

                if self.do_bn:
                    self.conv_bns.append(torch.nn.BatchNorm1d(self.channels[i]).to('cuda:'+str(devices[0])))
                    setattr(self, 'conv_bn%i' % i, self.conv_bns[i])

                if self.conv_afs[i] not in ['relu', 'tanh', 'hardtanh', 'leaky_relu']:
                    # TODO define custom not implemented activation function exception
                    raise Exception('Not implemented activation function: ' + self.conv_afs[i])
            else:
                if self.conv_paddings[i] is None:
                    self.convs.append(torch.nn.Conv1d(self.first_in_channel,
                                                      self.channels[i],
                                                      self.kernels[i],
                                                      stride=self.strides[i]).to('cuda:'+str(devices[1])))
                else:
                    self.convs.append(torch.nn.Conv1d(self.first_in_channel,
                                                      self.channels[i],
                                                      self.kernels[i],
                                                      stride=self.strides[i],
                                                      padding=conv_paddings[i]).to('cuda:'+str(devices[1])))
                setattr(self, 'conv%i' % i, self.convs[i])

                if self.do_bn:
                    self.conv_bns.append(torch.nn.BatchNorm1d(self.channels[i]).to('cuda:'+str(devices[1])))
                    setattr(self, 'conv_bn%i' % i, self.conv_bns[i])

                if self.conv_afs[i] not in ['relu', 'tanh', 'hardtanh', 'leaky_relu']:
                    # TODO define custom not implemented activation function exception
                    raise Exception('Not implemented activation function: ' + self.conv_afs[i])

        # Max pooling or average pooling over time would be used just like in the reference paper
        input_size_for_fcs = np.sum(conv_channels)

        # Fully connected layers
        for i in range(len(self.fc_hus)):
            # fc
            if not self.fcs:
                self.fcs.append(torch.nn.Linear(int(input_size_for_fcs), self.fc_hus[i]).to('cuda:'+str(devices[1])))
                setattr(self, 'fc%i' % i, self.fcs[i])
            else:
                self.fcs.append(torch.nn.Linear(self.fc_hus[i-1], self.fc_hus[i]).to('cuda:'+str(devices[1])))
                setattr(self, 'fc%i' % i, self.fcs[i])
            # bn
            if self.do_bn:
                self.fc_bns.append(torch.nn.BatchNorm1d(self.fc_hus[i]).to('cuda:'+str(devices[1])))
                setattr(self, 'fc_bn%i' % i, self.fc_bns[i])
            # do
            if self.do_dropout:
                self.fc_dos.append(torch.nn.Dropout(p=fc_dropout_rate).to('cuda:'+str(devices[1])))
                setattr(self, 'fc_do%i' % i, self.fc_dos[i])

            if self.fc_afs[i] not in ['relu', 'tanh', 'hardtanh', 'leaky_relu']:
                # TODO define custom not implemented activation function exception
                raise Exception('Not implemented activation function: ' + self.conv_afs[i])

        self.predict = torch.nn.Linear(self.fc_hus[-1], self.output_size).to('cuda:'+str(devices[1]))
        self.softmax = torch.nn.LogSoftmax(dim=1).to('cuda:'+str(devices[1]))

    def forward(self, x):
        x.to('cuda:'+str(self.devices[0]))
        # convs
        c = {}
        
        c[0] = self.convs[0](x)
        # Set activations
        if self.conv_afs[0] == 'relu':
            c[0] = torch.nn.ReLU()(c[0])
        elif self.conv_afs[0] == 'tanh':
            c[0] = torch.nn.Tanh()(c[0])
        elif self.conv_afs[0] == 'hardtanh':
            c[0] = torch.nn.Hardtanh()(c[0])
        elif self.conv_afs[0] == 'leaky_relu':
            c[0] = torch.nn.LeakyReLU()(c[0])

        if self.do_bn:
            c[0] = self.bns[0](c[0])

        c[1] = self.convs[1](x)
        # Set activations
        if self.conv_afs[1] == 'relu':
            c[1] = torch.nn.ReLU()(c[1])
        elif self.conv_afs[1] == 'tanh':
            c[1] = torch.nn.Tanh()(c[1])
        elif self.conv_afs[1] == 'hardtanh':
            c[1] = torch.nn.Hardtanh()(c[1])
        elif self.conv_afs[1] == 'leaky_relu':
            c[1] = torch.nn.LeakyReLU()(c[1])

        if self.do_bn:
            c[1] = self.bns[1](c[1])

        x.to('cuda:'+str(self.devices[1]))

        c[2] = self.convs[2](x)
        # Set activations
        if self.conv_afs[2] == 'relu':
            c[2] = torch.nn.ReLU()(c[2])
        elif self.conv_afs[2] == 'tanh':
            c[2] = torch.nn.Tanh()(c[2])
        elif self.conv_afs[2] == 'hardtanh':
            c[2] = torch.nn.Hardtanh()(c[2])
        elif self.conv_afs[2] == 'leaky_relu':
            c[2] = torch.nn.LeakyReLU()(c[2])

        if self.do_bn:
            c[2] = self.bns[2](c[2])

        x = c[0]
        x = torch.cat((x, c[1]), 0)

        x.to('cuda:'+str(self.devices[1]))

        x = torch.cat((x, c[2]), 0)

        if self.pooling_type == 'max':
            x = torch.max(x, 0)
        elif self.pooling_type == 'average':
            x = torch.mean(x, 0)

        # fc
        for i in range(len(self.fcs)):
            if i == 0:
                # like flatten in keras
                x = self.fcs[i](x)
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
                        self.poolings.append(torch.nn.AvgPool1d(self.pooling_kernels[i],
                                                                stride=self.pooling_strides[i]))
                    else:
                        self.poolings.append(torch.nn.AvgPool1d(self.pooling_kernels[i],
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
        x.to('cuda:0')
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
        output_width = 0
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
                output_width = (output_width - (self.poolings[i].kernel_size if isinstance(self.poolings[i].kernel_size, int) else self.poolings[i].kernel_size[0]) + 2 * (self.poolings[i].padding if isinstance(self.poolings[i].padding, int) else self.poolings[i].padding[0])) / (self.poolings[i].stride if isinstance(self.poolings[i].stride, int) else self.poolings[i].stride[0]) + 1

        return output_width * self.channels[-1]


