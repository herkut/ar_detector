import numpy as np
from preprocess.data_representation_preparer import DataRepresentationPreparer


def filter_out_nan(x, y):
    index_to_remove = y[y.isna() == True].index
    # index_to_remove = np.argwhere(np.isnan(y)).values

    xx = x.drop(index_to_remove, inplace=False)
    yy = y.drop(index_to_remove, inplace=False)

    # xx = np.delete(x, index_to_remove, axis=0)
    # yy = np.delete(y, index_to_remove, axis=0)

    return xx, yy


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