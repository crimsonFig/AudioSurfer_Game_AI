"""
Used for training or loading a LSTM model with the training data from audio_surfer.

Currently works without bugs.
May require some better training configuration.
May need a change to how the training data is given to the model during fitting, possibly -1 to initial shift...
    ... this may help improve accuracy as resulting prediction accuracy is boosted when prediction is shifted back.

Written by Triston Scallan
"""
from pandas import DataFrame, concat
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, LSTM
from matplotlib import pyplot as plt

import os
import numpy as np

CREATE = 0
LOAD = 1
MODE = LOAD         # config mode for creating a new model to train or loading one.

PATH_TO_RECORDS = 'training_data'
FILE_NAME = 'LSTM_train_data_session1.npy'
MODEL_PATH = 'models'
MODEL_V1_FILE = os.path.join(MODEL_PATH, 'audio_surfer_LSTM_V1_20B_50N.h5')
MODEL_V2_FILE = os.path.join(MODEL_PATH, 'audio_surfer_LSTM_V2_5B_35N.h5')
DEFAULT_MODEL = MODEL_V2_FILE

BATCH_SIZE = 5          # batch size of 5 appears to lead to best loss values
EPOCHS = 3000           # 3000 appears to be the point when training accuracy plateaus consistently
N_NEURONS = 35          # 35 neurons appeared to lead to better desired results
N_TRAIN_SAMPLES = 900
N_STEPS = 1             # given a single input row
N_PREDICTIONS = 1       # output a single prediction


def series_to_supervised(data, n_in=1, n_out=1, drop_nan=True) -> DataFrame:
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (j, i)) for j in data.columns]
    # predict sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data.shift(-i))
        if i == 0:
            names += [('%s(t)' % j) for j in data.columns]
        else:
            names += [('%s(t+%d)' % (j, i)) for j in data.columns]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if drop_nan:
        agg.dropna(inplace=True)
    return agg


def create_network() -> Sequential:
    # design network
    input_shape = (train_X.shape[1], train_X.shape[2])
    batch_shape = (BATCH_SIZE,) + input_shape
    _model = Sequential()
    # using relu, soft-max, and categorical cross entropy because we want only one of the output nodes active
    _model.add(
        LSTM(N_NEURONS, input_shape=input_shape, batch_input_shape=batch_shape,
             activation='relu', kernel_initializer='he_uniform', stateful=False))
    _model.add(Dense(3, activation='softmax'))
    _model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return _model


def perform_training(_model: Sequential, _epochs: int):
    # fit network
    return _model.fit(train_X, train_y, epochs=_epochs, batch_size=BATCH_SIZE, validation_data=(test_X, test_y),
                      shuffle=False)


def evaluate_fitness(_model: Sequential, _history):
    # evaluate (though this doesnt account for inverting the result)
    _, train_acc = _model.evaluate(train_X, train_y, BATCH_SIZE, verbose=2)
    _, test_acc = _model.evaluate(test_X, test_y, BATCH_SIZE, verbose=2)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    plt.subplot(211)
    plt.title('loss')
    plt.plot(_history.history['loss'], label='train')
    plt.plot(_history.history['val_loss'], label='test')
    plt.legend()
    plt.subplot(212)
    plt.title('acc')
    plt.plot(_history.history['acc'], label='train')
    plt.plot(_history.history['val_acc'], label='test')
    plt.legend()
    plt.show()


def predict(_model: Sequential, data_x: np.array) -> np.array:
    """
    Creates a prediction array based on input data.

    :param data_x: the data to make a prediction of
    :param _model: the LSTM model
    :return: a numpy array filled with the predictions of the given model and data
    """
    return _model.predict(data_x, BATCH_SIZE)


def eval_accuracy(_model: Sequential, data_x: np.array, data_y: np.array) -> np.float64:
    """
    Determines the accuracy of the predicted keys and the expected keys after inverting prediction.
    (by invert, this basically means to realign the predictions with respective expected result.)
    (accuracy is higher after performing the shift, may require a change to how training data is given during fitting)

    :param _model: the LSTM model
    :param data_x: the input data to analyze
    :param data_y: the expected key data
    :return: the accuracy as a float from 0.0 to 1.0
    """
    predicted_key = _model.predict(data_x, BATCH_SIZE)
    inverted_key_as_classes = np.roll(predicted_key.argmax(axis=1), -N_STEPS)
    expected_key_as_classes = data_y.argmax(axis=1)
    return np.equal(inverted_key_as_classes, expected_key_as_classes).sum() / data_y.shape[0]


if __name__ == '__main__':
    # load data
    record_array = np.load(os.path.join(PATH_TO_RECORDS, FILE_NAME))
    labels = ['block_x', 'block_y',
              'spike_x', 'spike_y',
              'ship_left', 'ship_center', 'ship_right',
              'key_<-', 'key_--', 'key_->']
    dataset = DataFrame(record_array, columns=labels, dtype=np.float32)
    lag_data = series_to_supervised(dataset, N_STEPS, N_PREDICTIONS)
    lag_data.drop(lag_data.columns[-10:-3], axis=1, inplace=True)

    # create train and test sets
    values = lag_data.values
    n_test_samples = (((dataset.shape[0] - N_TRAIN_SAMPLES - N_STEPS) // BATCH_SIZE) * BATCH_SIZE) + N_TRAIN_SAMPLES

    train_X, train_y = values[:N_TRAIN_SAMPLES, :-3], values[:N_TRAIN_SAMPLES, -3:]
    test_X, test_y = values[N_TRAIN_SAMPLES:n_test_samples, :-3], values[N_TRAIN_SAMPLES:n_test_samples, -3:]

    # reshape input into [samples, time-steps, features]
    train_X = train_X.reshape((train_X.shape[0], N_STEPS, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], N_STEPS, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    if MODE is CREATE:
        # define and fit the LSTM model
        model = create_network()
        history = perform_training(model, EPOCHS)
        evaluate_fitness(model, history)
        print('evaluated training accuracy: ', eval_accuracy(model, train_X, train_y))
        print('evaluated testing accuracy: ', eval_accuracy(model, test_X, test_y))
    elif MODE is LOAD:
        model = load_model(DEFAULT_MODEL)
        print('training data accuracy: ', eval_accuracy(model, train_X, train_y))
        print('test data accuracy: ', eval_accuracy(model, test_X, test_y))

    # # create evaluation models for printing
    # predicted_key = model.predict(train_X, BATCH_SIZE)
    # df = DataFrame(np.array([np.roll(predicted_key.argmax(axis=1), -N_STEPS), train_y.argmax(axis=1)]).T,
    #                columns=['Predicted_Key', 'Expected_Key'], dtype=np.int16)
