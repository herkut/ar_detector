import torch
import numpy as np


class FCBlock(torch.nn.Module):
    def __init__(self, device, feature_size, fc_hus, fc_activations, fc_dropout_rate, batch_normalization=True):
        self.device = device
        self.feature_size = feature_size
        # Fully connected layers
        self.fc_layers = []
        for i in range(len(self.fc_hus)):
            # fc
            if not self.fcs:
                tmp_layer = torch.nn.Linear(int(feature_size), fc_hus[i])
                setattr(self, 'fc%i' % i, tmp_layer)

            else:
                tmp_layer = torch.nn.Linear(self.fc_hus[i - 1], fc_hus[i])
                setattr(self, 'fc%i' % i, tmp_layer)
            self.fc_layers.append(tmp_layer)

            # bn
            if batch_normalization:
                tmp_layer = torch.nn.BatchNorm1d(self.fc_hus[i])
                setattr(self, 'fc_bn%i' % i, tmp_layer)
                self.fc_layers.append(tmp_layer)

            # do
            if fc_dropout_rate > 0:
                tmp_layer = torch.nn.Dropout(p=fc_dropout_rate)
                setattr(self, 'fc_do%i' % i, tmp_layer)
                self.fc_layers.append(tmp_layer)

            # Set activations
            if fc_activations[i] == 'relu':
                tmp_layer = torch.nn.ReLU()
            elif fc_activations[i] == 'tanh':
                tmp_layer = torch.nn.Tanh()
            elif fc_activations[i] == 'hardtanh':
                tmp_layer = torch.nn.Hardtanh()
            elif fc_activations[i] == 'leaky_relu':
                tmp_layer = torch.nn.LeakyReLU()
            else:
                # TODO define custom not implemented activation function exception
                raise Exception('Not implemented activation function: ' + self.activation_function)
            setattr(self, 'fc_activation%i' % i, tmp_layer)
            self.fc_layers.append(tmp_layer)

        self.block = torch.nn.Sequential(*self.fc_layers).to('cuda:' + str(self.device))


class ConvBlock(torch.nn.Module):
    def __init__(self, device, feature_size, incoming_channel, kernel, channel, padding, stride, activation_function, pooling='max', batch_normalization=True):
        """

        :param device:
        :param feature_size:
        :param kernel:
        :param channel:
        :param padding:
        :param stride:
        :param activation_function:
        """
        self.device = device
        self.feature_size = feature_size
        self.incoming_channel = incoming_channel
        self.kernel = kernel
        self.channel = channel
        self.padding = padding
        self.stride = stride
        self.activation_function = activation_function

        # Convolutional layers
        if padding is None:
            self.poolings_kernel_size = int((feature_size - kernel) / stride + 1)
        else:
            self.poolings_kernel_size = int((feature_size - kernel + 2 * padding) / stride + 1)

        print(str(self.poolings_kernel_size))

        # Conv0
        if self.padding is None:
            conv = torch.nn.Conv1d(self.incoming_channel,
                                   self.channel,
                                   self.kernel,
                                   stride=self.stride)
        else:
            conv = torch.nn.Conv1d(self.incoming_channel,
                                   self.channel,
                                   self.kernel,
                                   stride=self.stride,
                                   padding=self.padding)
        setattr(self, 'conv', self.conv)

        if self.batch_normalization:
            conv_bn = torch.nn.BatchNorm1d(self.channel)
            setattr(self, 'conv_bn', self.conv_bn)
        else:
            conv_bn = None

        # Set activations
        if self.activation_function == 'relu':
            af = torch.nn.ReLU()
        elif self.activation_function == 'tanh':
            af = torch.nn.Tanh()
        elif self.activation_function == 'hardtanh':
            af = torch.nn.Hardtanh()
        elif self.activation_function == 'leaky_relu':
            af = torch.nn.LeakyReLU()
        else:
            # TODO define custom not implemented activation function exception
            raise Exception('Not implemented activation function: ' + self.activation_function)
        setattr(self, 'activation', self.af)

        if pooling == 'max':
            pooling = torch.nn.MaxPool1d(kernel_size=self.poolings_kernel_size)
        else:
            pooling = torch.nn.AvgPool1d(kernel_size=self.poolings_kernel_size)
        setattr(self, 'pooling', self.pooling)

        if conv_bn is not None:
            self.block = torch.nn.Sequential(conv,
                                             conv_bn,
                                             af,
                                             pooling).to('cuda:' + str(self.device))
        else:
            self.block = torch.nn.Sequential(conv,
                                             af,
                                             pooling).to('cuda:' + str(self.device))


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

        self.kernels = conv_kernels  # kernel_sizes
        self.channels = conv_channels  # channels
        self.strides = conv_strides  # strides

        self.cbs = {}

        self.cbs[0] = ConvBlock(self.devices[0], feature_size, first_in_channel, conv_kernels[0], conv_channels[0],
                                conv_paddings[0], conv_strides[0], conv_activation_functions[0], pooling=pooling_type,
                                batch_normalization=batch_normalization)
        self.cbs[1] = ConvBlock(self.devices[1], feature_size, first_in_channel, conv_kernels[1], conv_channels[1],
                                conv_paddings[1], conv_strides[1], conv_activation_functions[1], pooling=pooling_type,
                                batch_normalization=batch_normalization)
        self.cbs[2] = ConvBlock(self.devices[2], feature_size, first_in_channel, conv_kernels[2], conv_channels[2],
                                conv_paddings[2], conv_strides[2], conv_activation_functions[2], pooling=pooling_type,
                                batch_normalization=batch_normalization)

        # Max pooling or average pooling over time would be used just like in the reference paper
        input_size_for_fcs = np.sum(conv_channels)

        self.fc = FCBlock(self.devices[3],
                          input_size_for_fcs,
                          fc_hidden_units,
                          fc_activation_functions,
                          fc_dropout_rate,
                          batch_normalization=batch_normalization)

        self.predict = torch.nn.Linear(fc_hidden_units[-1], self.output_size).to('cuda:' + str(devices[3]))
        self.softmax = torch.nn.LogSoftmax(dim=1).to('cuda:' + str(devices[3]))

    def forward(self, x):
        x0 = x.to('cuda:' + str(self.devices[0]), torch.float)
        x1 = x.to('cuda:' + str(self.devices[1]), torch.float)
        x2 = x.to('cuda:' + str(self.devices[2]), torch.float)

        # convs
        x0 = self.cbs[0].block(x0)
        x1 = self.cbs[1].block(x1)
        x2 = self.cbs[2].block(x2)

        x = torch.cat((x0.to('cuda:' + str(self.devices[3])),
                       x1.to('cuda:' + str(self.devices[3])),
                       x2.to('cuda:' + str(self.devices[3]))), 1)

        # batch, channels, res -> batch, res, channels
        x = x.transpose(1, 2)
        x = x.view(x.size(0), -1)

        # fc
        x = self.fc.block(x)

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
                    self.convs.append(torch.nn.Conv1d(self.channels[i - 1],
                                                      self.channels[i],
                                                      self.kernels[i],
                                                      stride=self.strides[i]))
                else:
                    self.convs.append(torch.nn.Conv1d(self.channels[i - 1],
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
                # if pooling is overtime
                if self.pooling_kernels[i] == "overtime":
                    if self.pooling_type == 'max':
                        if self.pooling_paddings[i] is None:
                            self.poolings.append(torch.nn.MaxPool1d(self._calculate_kernel_size_for_pooling_overtime_at(i),
                                                                    stride=self.pooling_strides[i]))
                        else:
                            self.poolings.append(torch.nn.MaxPool1d(self._calculate_kernel_size_for_pooling_overtime_at(i),
                                                                    stride=self.pooling_strides[i],
                                                                    padding=self.pooling_paddings[i]))
                        setattr(self, 'pool_over_time%i' % i, self.poolings[i])
                    elif self.pooling_type == 'average':
                        if self.pooling_paddings[i] is None:
                            self.poolings.append(torch.nn.AvgPool1d(self._calculate_kernel_size_for_pooling_overtime_at(i),
                                                                    stride=self.pooling_strides[i]))
                        else:
                            self.poolings.append(torch.nn.AvgPool1d(self._calculate_kernel_size_for_pooling_overtime_at(i),
                                                                    stride=self.pooling_strides[i],
                                                                    padding=self.pooling_paddings[i]))
                        setattr(self, 'pool_over_time%i' % i, self.poolings[i])
                else:
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
                # else
                # TODO raise unknown pooling method
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
                self.fcs.append(torch.nn.Linear(self.fc_hus[i - 1], self.fc_hus[i]))
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
                x = self.fcs[i](x.view(-1, x.shape[1] * x.shape[2]))
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
                    output_width = (self.feature_size - self.kernels[i] + 2 * self.conv_paddings[i]) / self.strides[
                        i] + 1
            else:
                if self.conv_paddings[i] is None:
                    output_width = (output_width - self.kernels[i] + 2 * 0) / self.strides[i] + 1
                else:
                    output_width = (output_width - self.kernels[i] + 2 * self.conv_paddings[i]) / self.strides[i] + 1
            # pooling
            if self.poolings[i] is not None:
                output_width = (output_width -
                                (self.poolings[i].kernel_size if isinstance(self.poolings[i].kernel_size, int) else
                                    self.poolings[i].kernel_size[0])
                                + 2 * (self.poolings[i].padding if isinstance(self.poolings[i].padding, int) else
                                           self.poolings[i].padding[0])) \
                               / (self.poolings[i].stride if isinstance(self.poolings[i].stride, int) else
                                     self.poolings[i].stride[0]) \
                               + 1

        return int(output_width * self.channels[-1])

    def _calculate_kernel_size_for_pooling_overtime_at(self, level):
        output_width = 0
        for i in range(level):
            if i == 0:
                if self.conv_paddings[i] is None:
                    output_width = (self.feature_size - self.kernels[i] + 2 * 0) / self.strides[i] + 1
                else:
                    output_width = (self.feature_size - self.kernels[i] + 2 * self.conv_paddings[i]) / self.strides[
                        i] + 1
            else:
                if self.conv_paddings[i] is None:
                    output_width = (output_width - self.kernels[i] + 2 * 0) / self.strides[i] + 1
                else:
                    output_width = (output_width - self.kernels[i] + 2 * self.conv_paddings[i]) / self.strides[i] + 1
            # pooling
            if self.poolings[i] is not None:
                output_width = (output_width - (
                    self.poolings[i].kernel_size if isinstance(self.poolings[i].kernel_size, int) else
                    self.poolings[i].kernel_size[0]) + 2 * (
                                    self.poolings[i].padding if isinstance(self.poolings[i].padding, int) else
                                    self.poolings[i].padding[0])) / (
                                   self.poolings[i].stride if isinstance(self.poolings[i].stride, int) else
                                   self.poolings[i].stride[0]) + 1

        if level == 0:
            if self.conv_paddings[level] is None:
                output_width = (self.feature_size - self.kernels[level] + 2 * 0) / self.strides[level] + 1
            else:
                output_width = (self.feature_size - self.kernels[level] + 2 * self.conv_paddings[level]) / self.strides[level] + 1
        else:
            if self.conv_paddings[level] is None:
                output_width = (output_width - self.kernels[level] + 2 * 0) / self.strides[level] + 1
            else:
                output_width = (output_width - self.kernels[level] + 2 * self.conv_paddings[level]) / self.strides[level] + 1

        return int(output_width)
