import os

import torch

from config import Config
from models.base_ar_detector import BaseARDetector
from preprocess.cnn_dataset import ARCNNDataset
from preprocess.feature_label_preparer import FeatureLabelPreparer
from utils.helper_functions import get_index_to_remove, get_k_fold
import numpy as np


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

            if self.poolings[i] is not None:
                x = self.poolings[i](x)

        # fc
        for i in range(len(self.fcs)):
            if i ==0:
                # like flatten
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
                output_width = (self.feature_size - self.kernels[i] + 2 * 0) / self.strides[i] + 1
            else:
                output_width = (output_width - self.kernels[i] + 2 * 0) / self.strides[i] + 1
            # pooling
            if self.poolings[i] is not None:
                output_width = (output_width - self.poolings[i].kernel_size + 2 * self.poolings[i].padding) / self.poolings[i].stride + 1

        return output_width * self.channels[-1]


class ARDetectorCNN(BaseARDetector):
    def __init__(self, feature_size, first_in_channel, hyperparameters, antibiotic_name=None, model_name='cnn', class_weights=None):
        # self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_usages = np.zeros(gpu_count)
            for i in range(gpu_count):
                gpu_usages[i] = torch.cuda.memory_allocated(i)
            least_used_gpu = np.argmin(gpu_usages)
            self._device = torch.device("cuda:" + str(least_used_gpu))
        else:
            self._device = "cpu"

    def _initialize_model(self, device, feature_size, first_in_channel, class_weights, hyperparameters):
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
                          hyperparameters['conv_activation_functions'],
                          hyperparameters['pooling_kernels'],
                          hyperparameters['pooling_strides'],
                          hyperparameters['fc_hidden_units'],
                          hyperparameters['fc_activation_functions'],
                          hyperparameters['fc_dropout'],
                          batch_normalization=True,
                          pooling_type=hyperparameters['pooling_types'])
        model.to(device)

        if class_weights is not None:
            criterion = torch.nn.NLLLoss(reduction='mean', weight=torch.from_numpy(class_weights).to(device))
        else:
            criterion = torch.nn.NLLLoss(reduction='mean')

        if hyperparameters['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(conv_net.parameters(), lr=learning_rate)
        elif hyperparameters['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(conv_net.parameters(), lr=learning_rate)
        elif hyperparameters['optimizer'] == 'Adamax':
            optimizer = torch.optim.Adamax(conv_net.parameters(), lr=learning_rate)
        elif hyperparameters['optimizer'] == 'RMSProp':
            optimizer = torch.optim.RMSProp(conv_net.parameters(), lr=learning_rate)
        else:
            raise Exception('Not implemented optimizer: ' + hyperparameters['optimizer'])

        return model, criterion, optimizer

    def load_model(self):
        pass

    def save_model(self):
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
    feature_size = 37733
    first_in_channel = 5
    output_size = 2
    optimizer = 'Adam'
    learning_rate = 0.001

    configuration_file = '/home/herkut/Desktop/ar_detector/configurations/conf.yml'
    raw = open(configuration_file)
    Config.initialize_configurations(raw)

    raw_label_matrix = FeatureLabelPreparer.get_labels_from_file(os.path.join(Config.dataset_directory,
                                                                              'sorted_labels_dataset-ii.csv'))

    target_drug = Config.target_drugs[0]

    non_existing = []
    predefined_file_to_remove = ['8316-09', 'NL041']

    index_to_remove = get_index_to_remove(raw_label_matrix[target_drug])

    for ne in predefined_file_to_remove:
        if ne not in index_to_remove:
            non_existing.append(ne)

    raw_label_matrix.drop(index_to_remove, inplace=True)
    raw_label_matrix.drop(non_existing, inplace=True)

    idx = raw_label_matrix.index
    labels = raw_label_matrix[target_drug].values

    unique, counts = np.unique(labels, return_counts=True)

    # class_weights = {0: counts[1] / (counts[0] + counts[1]), 1: counts[0] / (counts[0] + counts[1])}
    class_weights = {0: np.max(counts) / counts[0], 1: np.max(counts) / counts[1]}
    class_weights = np.array(list(class_weights.items()), dtype=np.float32)[:, 1]
    ##############################################
    #                                            #
    #       Convolutional Neural Network         #
    #                                            #
    ##############################################
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
    criterion = torch.nn.NLLLoss(reduction='mean', weight=torch.from_numpy(class_weights))
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(conv_net.parameters(), lr=learning_rate)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(conv_net.parameters(), lr=learning_rate)
    elif optimizer == 'Adamax':
        optimizer = torch.optim.Adamax(conv_net.parameters(), lr=learning_rate)
    elif optimizer == 'RMSProp':
        optimizer = torch.optim.RMSProp(conv_net.parameters(), lr=learning_rate)
    else:
        raise Exception('Not implemented optimizer: ' + optimizer)

    print(conv_net)

    cv = get_k_fold(10)

    for train_index, test_index in cv.split(idx, labels):
        tr_dataset = ARCNNDataset(idx[train_index], labels[train_index], target_drug)
        tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=64)
        te_dataset = ARCNNDataset(idx[test_index], labels[test_index], target_drug)
        te_dataloader = torch.utils.data.DataLoader(te_dataset, batch_size=64)

        training_results = None
        training_loss = 0.0
        for epoch in range(20):
            for i, data in enumerate(tr_dataloader):
                inputs, labels = data
                # initialization of gradients
                optimizer.zero_grad()
                # Forward propagation
                y_hat = conv_net(inputs.float())
                pred = torch.argmax(y_hat, dim=1)
                # Computation of cost function
                cost = criterion(y_hat, labels)
                # Back propagation
                cost.backward()
                # Update parameters
                optimizer.step()

                training_loss += cost
