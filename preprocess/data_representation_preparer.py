import numpy as np


class DataRepresentationPreparer:

    @staticmethod
    def update_feature_matrix_with_tf_idf(feature_matrix):
        # Compute term frequencies
        tf = feature_matrix.divide((1 + np.sum(feature_matrix, axis=1)), axis=0)

        # Compute inverse document frequencies
        idf = np.log2(len(tf) / feature_matrix[feature_matrix > 0].count())

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
        rf = np.log2(2 + a.divide(1+b))

        tfrf = np.multiply(tf, rf.to_frame().T)

        return tfrf

    # k parameter is taken from Ozlem's paper named 'Statistical representation models for mutation information within genomic data'
    @staticmethod
    def update_feature_matrix_with_bm25_tf_idf(feature_matrix):
        k = 0.8
        # Compute term frequencies
        tf = feature_matrix.divide((1 + np.sum(feature_matrix, axis=1)), axis=0)
        bm25tf = ((k + 1) * tf).divide(k + tf)

        # Compute inverse document frequencies
        idf = np.log2(len(tf) / feature_matrix[feature_matrix > 0].count())

        bm25tfidf = np.multiply(bm25tf, idf.to_frame().T)

        return bm25tfidf

    # k parameter is taken from Ozlem's paper named 'Statistical representation models for mutation information within genomic data'
    @staticmethod
    def update_feature_matrix_with_bm25_tf_rf(feature_matrix, labels):
        k = 0.8
        # Compute term frequencies
        tf = feature_matrix.divide((1 + np.sum(feature_matrix, axis=1)), axis=0)
        bm25tf = ((k + 1) * tf).divide(k + tf)

        # Compute relevance frequency
        l_1 = labels[labels == 1].index
        l_0 = labels[labels == 0].index

        a = feature_matrix.loc[l_1, :][feature_matrix == 1].count()
        b = feature_matrix.loc[l_0, :][feature_matrix == 1].count()

        # Add 1 to avoid 0 division
        rf = np.log2(2 + a.divide(1 + b))

        bm25tfrf = np.multiply(bm25tf, rf.to_frame().T)

        return bm25tfrf