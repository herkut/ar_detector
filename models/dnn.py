from keras import Sequential
from keras.layers import Dense, BatchNormalization


class DNN:
    def __init__(self, feature_size, output_size, hidden_units_arr, activation_functions_arr, batch_normalization_required=False, optimizer='adam'):
        self.feature_size = feature_size
        self.output_size = output_size
        self.hidden_units_arr = hidden_units_arr
        self.activation_functions_arr = activation_functions_arr
        self.batch_normalization_required = batch_normalization_required
        self.optimizer = optimizer

        self.model = self.create_model()

    def create_model(self, hidden_units_arr):
        model = Sequential()

        # Initialize first hidden layer
        model.add(Dense(hidden_units_arr[0], input_dim=self.feature_size, activation=self.activation_functions_arr[0]))
        if self.batch_normalization_required:
            model.add(BatchNormalization())
        for i in range(1, len(hidden_units_arr)):
            model.add(Dense(hidden_units_arr[i], activation=self.activation_functions_arr[i]))
            if self.batch_normalization_required:
                model.add(BatchNormalization())
        # Initialize output layer which is multilabel classification in our problem
        model.add(Dense(self.output_size, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        return model

