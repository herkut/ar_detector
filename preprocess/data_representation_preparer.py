import numpy as np


class DataRepresentationPreparer:

    @staticmethod
    def update_feature_matrix_with_tf_idf(feature_matrix):
        # Compute term frequencies
        tf = feature_matrix.divide((1 + np.sum(feature_matrix, axis=1)), axis=0)

        # Compute inverse document frequencies
        idf = np.log(len(tf) / feature_matrix[feature_matrix > 0].count())

        tfidf = np.multiply(tf, idf.to_frame().T)

        return tfidf

    @staticmethod
    def update_feature_matrix_with_tf_rf(feature_matrix, labels):
        # Compute term frequencies
        tf = feature_matrix.divide((1 + np.sum(feature_matrix, axis=1)), axis=0)

        # Compute relevance frequency
        l_1 = labels[labels == 1].index
        l_0 = labels[labels == 0].index

        a = feature_matrix.loc[l_1, :][feature_matrix == 1].count()
        b = feature_matrix.loc[l_0, :][feature_matrix == 1].count()

        # Add 1 to avoid 0 division
        rf = np.log(2 + a.divide(1+b))

        tfrf = np.multiply(tf, rf.to_frame().T)

        return tfrf

    # TODO define functions for bm25tfidf and bm25tfrf
