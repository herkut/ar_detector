import json
import timeit

import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

from deprecated.tensorflow_models.ar_detector_dnn import Dnn1D
from deprecated.tensorflow_models.ar_detector_dnn import Dnn2D
from deprecated.tensorflow_models.ar_detector_dnn import DnnND

from preprocess.data_representation_preparer import DataRepresentationPreparer


######################################################################
first_line_drugs = ['Isoniazid', 'Rifampicin', 'Ethambutol', 'Pyrazinamide']
target_drugs = ['Isoniazid', 'Rifampicin', 'Ethambutol', 'Pyrazinamide', 'Streptomycin', 'Ofloxacin', 'Amikacin', 'Ciprofloxacin', 'Moxifloxacin', 'Capreomycin', 'Kanamycin']
label_tags = 'phenotype'
directory_containing_indexes = '/run/media/herkut/herkut/TB_genomes/dataset-1-train_test_indexes/'
directory_containing_model_logs = '/run/media/herkut/hdd-1/TB_genomes/ar_detector_results/logs/'
TEST_SIZE = 0.2
######################################################################


def get_k_fold_validation_indices(k, X, y):
    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=False)
    return skf.split(X, y)


class TensorflowModelManager:
    def __init__(self, models, result_directory, data_representation='binary'):
        self.data_representation = data_representation
        self.result_directory = result_directory
        self.feature_selection = None
        # Set which models would be trained
        self.models_arr = models.split(',')

    def train_and_test_models(self, raw_feature_matrix, raw_labels):
        for model in self.models_arr:
            # DNN
            if model.startswith('dnn-'):
                for i in range(len(target_drugs)):
                    x, y = self.filter_out_nan(raw_feature_matrix, raw_labels[target_drugs[i]])

                    tr_indexes = np.genfromtxt(directory_containing_indexes + target_drugs[i] + '_tr_indices.csv',
                                               delimiter=' ',
                                               dtype=np.int32)

                    te_indexes = np.genfromtxt(directory_containing_indexes + target_drugs[i] + '_te_indices.csv',
                                               delimiter=' ',
                                               dtype=np.int32)

                    # Random state is used to make train and test split the same on each iteration
                    if self.data_representation == 'tfidf':
                        x = DataRepresentationPreparer.update_feature_matrix_with_tf_idf(x)
                    elif self.data_representation == 'tfrf':
                        x = DataRepresentationPreparer.update_feature_matrix_with_tf_rf(x, y)
                    elif self.data_representation == 'bm25tfidf':
                        x = DataRepresentationPreparer.update_feature_matrix_with_bm25_tf_idf(x)
                    elif self.data_representation == 'bm25tfrf':
                        x = DataRepresentationPreparer.update_feature_matrix_with_bm25_tf_rf(x, y)
                    else:
                        # Assumed binary data representation would be used
                        pass

                    x_tr = x.loc[tr_indexes].values
                    y_tr = y.loc[tr_indexes].values
                    x_te = x.loc[te_indexes].values
                    y_te = y.loc[te_indexes].values

                    x = x.values
                    y = y.values

                    class_weights = np.zeros(2)

                    unique, counts = np.unique(y, return_counts=True)

                    class_weights[0] = counts[1] / (counts[0] + counts[1])
                    class_weights[1] = counts[0] / (counts[0] + counts[1])

                    print('Class weights: ' + str(class_weights))

                    print("For the antibiotic " + target_drugs[i])
                    print("Size of training dataset " + str(np.shape(x_tr)))
                    print("Size of test dataset " + str(np.shape(x_te)))

                    model_info = model.split('-')
                    model_dim = model_info[1].split('d')[0]

                    if int(model_dim) == 1:
                        ar_detector = Dnn1D(x_tr.shape[1], 2)
                    elif int(model_dim) == 2:
                        ar_detector = Dnn2D(x_tr.shape[1], 2)
                    else:
                        ar_detector = DnnND(x_tr.shape[1], 2, model_dim)

                    ar_detector.set_antibiotic_name(target_drugs[i])
                    self.train_dnn_with_gridsearch(ar_detector, x_tr, y_tr, class_weights)
                    # self.train_dnn_with_randomsearch(ar_detector)

    def filter_out_nan(self, x, y):
        index_to_remove = y[y.isna() == True].index

        xx = x.drop(index_to_remove, inplace=False)
        yy = y.drop(index_to_remove, inplace=False)

        return xx, yy

    def train_dnn_with_gridsearch(self, ar_detector, x_tr, y_tr, class_weights):
        # Optimizers to be tried are selected according to Karpathy's following blog page: https://medium.com/@karpathy/a-peek-at-trends-in-machine-learning-ab8a1085a106

        # hidden units and activation functions elements must be the same sized because they would create a hidden layer together
        # Single layer neural network
        hidden_units=[[128], [1024], [4096]]

        #activation_functions=[['relu'], ['leaky_relu'], ['elu'], ['tanh']]
        activation_functions = [['relu'], ['leaky_relu']]

        """
        # Two layers neural network
        hidden_units = [[512, 128], [512, 512], [128,512]]

        activation_functions = [['leaky_relu', 'leaky_relu'], ['relu', 'relu']]
        """
        learning_rates = [0.1, 0.01, 0.001]

        #optimizers=['RMSprop', 'Adam', 'Adagrad', 'Adadelta']
        optimizers = ['Adam']

        dropout_rates=[0.0, 0.25, 0.5]

        kfold_indices = get_k_fold_validation_indices(5, x_tr, y_tr)

        # Convert labels into one hot encoded format
        y_tr_one_hot_encoded = np.zeros((len(y_tr), 2))
        for i in range(len(y_tr)):
            if y_tr[i] == 1:
                y_tr_one_hot_encoded[i, 1] = 1
            else:
                y_tr_one_hot_encoded[i, 0] = 1

        grid_search_results = {}
        grid_search_results['parameters'] = []
        grid_search_results['results'] = []
        grid_search_results['rank'] = []

        # Grid search on different architecture
        for lr in learning_rates:
            for dr in dropout_rates:
                for optimizer_method in optimizers:
                    for hu in hidden_units:
                        for af in activation_functions:
                            tmp_parameters = {}
                            tmp_parameters['learning_rate'] = lr
                            tmp_parameters['dropout_rate'] = dr
                            tmp_parameters['optimizer'] = optimizer_method
                            tmp_parameters['hidden_units'] = hu
                            tmp_parameters['activation_functions'] = af

                            grid_search_results['parameters'].append(tmp_parameters)
                            print(tmp_parameters)

                            # 5-fold cross validation
                            fold_index = 0
                            result = {}
                            result['training_accuracies'] = []
                            result['validation_accuracies'] = []
                            result['training_costs'] = []
                            result['validation_costs'] = []

                            # would be used as a lower limit for early stopping, it is used to prevent stopping at first epochs
                            min_iteration = 100

                            kfold_indices = get_k_fold_validation_indices(5, x_tr, y_tr)

                            start = timeit.default_timer()

                            for train_indices, validation_indices in kfold_indices:
                                # Reset tensorflow graph
                                tf.reset_default_graph()

                                ar_detector.create_model(hidden_units=hu,
                                                         activation_functions=af,
                                                         dropout_rate=dr,
                                                         learning_rate=lr,
                                                         optimizer_method=optimizer_method)

                                print('Split ' + str(fold_index))
                                with tf.Session() as session:
                                    # Run the global variable initializer to initialize all variables and layers of the neural network
                                    session.run(tf.global_variables_initializer())

                                    # create log file writer to record training progress.
                                    training_writer = tf.summary.FileWriter(directory_containing_model_logs + 'training', session.graph)
                                    testing_writer = tf.summary.FileWriter(directory_containing_model_logs + 'testing', session.graph)

                                    for epoch in range(2000):
                                        # Feed in the training data and do one step of neural network training
                                        ar_detector.train_model(session, x_tr[train_indices], y_tr_one_hot_encoded[train_indices], class_weights)

                                        """
                                        bs = 32 
                                        minibatch_count = int(x_tr.shape[0]/bs)
                                        for i in range(minibatch_count):
                                          session.run(optimizer, feed_dict={X:x_tr[i*bs:i*bs+bs,:], Y:y_tr_one_hot_encoded[i*bs:i*bs+bs,:], WO: class_weights})
                                        session.run(optimizer, feed_dict={X:x_tr[minibatch_count*bs:,:], Y:y_tr_one_hot_encoded[minibatch_count*bs:,:], WO: class_weights})
                                        """

                                        # TODO Convert early stopping metric from min of cost to max of accuracy
                                        tr_cost = ar_detector.get_cost(session, x_tr[train_indices], y_tr_one_hot_encoded[train_indices], class_weights)
                                        val_cost = ar_detector.get_cost(session, x_tr[validation_indices], y_tr_one_hot_encoded[validation_indices], class_weights)
                                        if epoch == 0:
                                            best_val_cost = val_cost
                                            best_val_cost_index = epoch
                                            ar_detector.saver.save(session, self.result_directory + '/tmp/best_model.ckpt')
                                        if val_cost < best_val_cost:
                                            best_val_cost = val_cost
                                            best_val_cost_index = epoch
                                            ar_detector.saver.save(session, self.result_directory + '/tmp/best_model.ckpt')

                                        tr_accuracy = ar_detector.get_accuracy(session, x_tr[train_indices], y_tr_one_hot_encoded[train_indices], class_weights)
                                        val_accuracy = ar_detector.get_accuracy(session, x_tr[validation_indices], y_tr_one_hot_encoded[validation_indices], class_weights)

                                        # Early stopping done like in bengio's random search paper
                                        if epoch / 2 > best_val_cost_index and epoch > min_iteration:
                                            print('Early stopping at epoch: ' + str(epoch))
                                            print(epoch, tr_cost, val_cost, tr_accuracy, val_accuracy)

                                            tr_summary = ar_detector.get_summary(session, x_tr[train_indices], y_tr_one_hot_encoded[train_indices], class_weights)
                                            val_summary = ar_detector.get_summary(session, x_tr[validation_indices], y_tr_one_hot_encoded[validation_indices], class_weights)

                                            training_writer.add_summary(tr_summary, epoch)
                                            testing_writer.add_summary(val_summary, epoch)
                                            break

                                        # Every 50 training steps, log our progress
                                        if epoch % 50 == 0:
                                            print(epoch, tr_cost, val_cost, tr_accuracy, val_accuracy)

                                            tr_summary = ar_detector.get_summary(session, x_tr[train_indices], y_tr_one_hot_encoded[train_indices], class_weights)
                                            val_summary = ar_detector.get_summary(session, x_tr[validation_indices], y_tr_one_hot_encoded[validation_indices], class_weights)

                                            training_writer.add_summary(tr_summary, epoch)
                                            testing_writer.add_summary(val_summary, epoch)

                                    #############################
                                    #                           #
                                    # Training is now complete! #
                                    #                           #
                                    #############################
                                    print("Training is complete!\n")

                                    ar_detector.saver.restore(session, self.result_directory + "/tmp/best_model.ckpt")

                                    final_tr_accuracy = ar_detector.get_accuracy(session, x_tr[train_indices], y_tr_one_hot_encoded[train_indices], class_weights)
                                    final_val_accuracy = ar_detector.get_accuracy(session, x_tr[validation_indices], y_tr_one_hot_encoded[validation_indices], class_weights)

                                    result['training_accuracies'].append(final_tr_accuracy)
                                    result['validation_accuracies'].append(final_val_accuracy)

                                    final_tr_cost = ar_detector.get_cost(session, x_tr[train_indices], y_tr_one_hot_encoded[train_indices], class_weights)
                                    final_val_cost = ar_detector.get_cost(session, x_tr[validation_indices], y_tr_one_hot_encoded[validation_indices], class_weights)

                                    result['training_costs'].append(final_tr_cost)
                                    result['validation_costs'].append(final_val_cost)

                                    print("Final Training Accuracy: {}, Cost: {}".format(final_tr_accuracy, final_tr_cost))
                                    print("Final Validation Accuracy: {}, Cost: {}\n".format(final_val_accuracy, final_val_cost))

                                    training_writer.close()
                                    testing_writer.close()
                                    session.close()
                                fold_index = fold_index + 1
                                ar_detector.clear_model()

                            result['mean_training_accuracy'] = np.mean(result['training_accuracies'])
                            result['std_training_accuracy'] = np.std(result['training_accuracies'])
                            result['mean_validation_accuracy'] = np.mean(result['validation_accuracies'])
                            result['std_validation_accuracy'] = np.std(result['validation_accuracies'])

                            result['mean_training_cost'] = np.mean(result['training_costs'])
                            result['std_training_cost'] = np.std(result['training_costs'])
                            result['mean_validation_cost'] = np.mean(result['validation_costs'])
                            result['std_validation_cost'] = np.std(result['validation_costs'])
                            grid_search_results['results'].append(result)

                            stop = timeit.default_timer()

                            print('Cross validation took ' + str(stop - start) + ' seconds for ' + str(tmp_parameters))

        with open(
                self.result_directory + '/grid_search_scores/' + self.feature_selection + self.data_representation + ar_detector.antibiotic_name + '.json',
                'w') as fp:
            json.dump(grid_search_results, fp)

    def test_dnn(self, ar_detector, x_te, y_te):
        print('Test ' + str(x_te.shape) + ' ' + str(y_te.shape))

        ar_detector.initialize_test_dataset(x_te, y_te)
        ar_detector.load_model()
        ar_detector.test_model()

    def set_feature_selection(self, feature_selection):
        self.feature_selection = feature_selection

    def extract_results(self, session, ar_detector, x, y_one_hot_encoded, class_weights):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        results = {}

        predictions = ar_detector.get_predictions(session, x, y_one_hot_encoded, class_weights)

        for i in range(len(y_one_hot_encoded)):
            if np.argmax(y_one_hot_encoded[i]) == 1 and np.argmax(predictions[i]) == 1:
                tp = tp + 1
            elif np.argmax(y_one_hot_encoded[i]) == 1 and np.argmax(predictions[i]) == 0:
                fn = fn + 1
            elif np.argmax(y_one_hot_encoded[i]) == 0 and np.argmax(predictions[i]) == 0:
                tn = tn + 1
            elif np.argmax(y_one_hot_encoded[i]) == 0 and np.argmax(predictions[i]) == 1:
                fp = fp + 1

        print('Recall/Sensitivity: ' + str(tp / (tp + fn)))
        print('Precision: ' + str(tp / (tp + fp)))
        print('Specificity: ' + str(tn / (tn + fp)))
        # F1 = 2 * (precision * recall) / (precision + recall)
        print('F1 score: ' + str(2 * ((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn)))))

        results['tp'] = tp
        results['tn'] = tn
        results['fp'] = fp
        results['fn'] = fn
        results['recall'] = tp / (tp + fn)
        results['precision'] = tp / (tp + fp)
        results['specificity'] = tn / (tn + fp)
        results['f1_score'] = 2 * ((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn)))

        return results
