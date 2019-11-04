import numpy as np
from sklearn.model_selection import StratifiedKFold

from preprocess.data_representation_preparer import DataRepresentationPreparer


def filter_out_nan(x, y):
    index_to_remove = y[y.isna() == True].index

    xx = x.drop(index_to_remove, inplace=False)
    yy = y.drop(index_to_remove, inplace=False)

    return xx, yy


def get_index_to_remove(y):
    index_to_remove = y[y.isna()].index

    return index_to_remove


def conduct_data_preprocessing(raw_feature_matrix, raw_labels, data_representation):
    x, y = filter_out_nan(raw_feature_matrix, raw_labels)

    # Random state is used to make train and test split the same on each iteration
    if data_representation == 'tfidf':
        x = DataRepresentationPreparer.update_feature_matrix_with_tf_idf(x)
    elif data_representation == 'tfrf':
        x = DataRepresentationPreparer.update_feature_matrix_with_tf_rf(x, y)
    elif data_representation == 'bm25tfidf':
        x = DataRepresentationPreparer.update_feature_matrix_with_bm25_tf_idf(x)
    elif data_representation == 'bm25tfrf':
        x = DataRepresentationPreparer.update_feature_matrix_with_bm25_tf_rf(x, y)
    else:
        # Assumed binary data representation would be used
        pass

    return x, y


def get_k_fold_indices(k, X, y):
    skf = StratifiedKFold(n_splits=k, random_state=0, shuffle=False)
    return skf.split(X, y)


def get_k_fold(k):
    skf = StratifiedKFold(n_splits=k, random_state=0, shuffle=False)
    return skf


def create_hyperparameter_space_for_cnn(param_grid):
    """
    conv_kernels,
    conv_channels,
    conv_strides,
    conv_activation_functions,
    pooling_kernels,
    pooling_strides,
    fc_hidden_units,
    fc_activation_functions,
    fc_dropout_rates,
    batch_normalization=False,
    pooling_type=None
    """
    hyperparameter_space = []
    for bs in param_grid['batch_sizes']:
        for optimizer_param in param_grid['optimizers']:
            for lr in param_grid['learning_rates']:
                for conv_kernels in param_grid['conv_kernels']:
                    for conv_channels in param_grid['conv_channels']:
                        for conv_strides in param_grid['conv_strides']:
                            for conv_afs in param_grid['conv_activation_functions']:
                                for pooling_kernels in param_grid['pooling_kernels']:
                                    for pooling_strides in param_grid['pooling_strides']:
                                        for fc_hus in param_grid['fc_hidden_units']:
                                            for fc_afs in param_grid['fc_activation_functions']:
                                                for fc_do in param_grid['fc_dropout_rates']:
                                                    for pooling_type in param_grid['pooling_types']:
                                                        grid = {}
                                                        grid['batch_size'] = bs
                                                        grid['optimizer'] = optimizer_param
                                                        grid['learning_rate'] = lr
                                                        grid['conv_kernels'] = conv_kernels
                                                        grid['conv_channels'] = conv_channels
                                                        grid['conv_strides'] = conv_strides
                                                        grid['conv_activation_functions'] = conv_afs
                                                        grid['pooling_kernels'] = pooling_kernels
                                                        grid['pooling_strides'] = pooling_strides
                                                        grid['fc_hidden_units'] = fc_hus
                                                        grid['fc_activation_functions'] = fc_afs
                                                        grid['fc_dropout_rate'] = fc_do
                                                        grid['pooling_type'] = pooling_type
                                                        hyperparameter_space.append(grid)

    return hyperparameter_space


def create_hyperparameter_space(param_grid):
    hyperparameter_space = []
    for bs in param_grid['batch_sizes']:
        for optimizer_param in param_grid['optimizers']:
            for lr in param_grid['learning_rates']:
                for hu in param_grid['hidden_units']:
                    for af in param_grid['activation_functions']:
                        for dr in param_grid['dropout_rates']:
                            grid = {}
                            grid['batch_size'] = bs
                            grid['optimizer'] = optimizer_param
                            grid['learning_rate'] = lr
                            grid['hidden_units'] = hu
                            grid['activation_functions'] = af
                            grid['dropout_rate'] = dr

                            hyperparameter_space.append(grid)

    return hyperparameter_space
