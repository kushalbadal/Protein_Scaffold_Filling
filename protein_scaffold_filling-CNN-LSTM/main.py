
# THIS IS IMPORTANT TO ACHEIVE REPRODUCIBILITY WITH TENSORFLOW. MUST HAPPEN BEFORE TENSORFLOW IMPORT
import os

# SHOULD HAVE THIS ENVIRONMENT VARIABLE SET BEFORE PYTHON EVEN BEGINS EXECUTION
os.environ['PYTHONHASHSEED']=str(1)

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers 

import random
import json
import csv

import numpy as np

from keras.utils import np_utils
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import utility as ut

os.system('color')

# Use monospace to make formatting cleaner
plt.rcParams.update({'font.family':'monospace'})

# =============================================================================
# Globals here
# =============================================================================

RESULTS_PATH = os.path.join(os.getcwd(), "results")
DATAPATH = os.path.join(os.getcwd(), "data")

# Model parameters
KMER_LENGTH = 5
MAX_EPOCH_LENGTH = 200
FIRST_LSTM_LAYER = 256
NUM_FILTERS = 128
FILTER_SIZE = 3
FIRST_DENSE_LAYER = 128
SECOND_DENSE_LAYER = 64
COST_FUNC = "categorical_crossentropy"
OPTIMIZER = "adam"
OUTPUT_ACTIVATION_FUNC = "softmax"
HIDDEN_LAYER_ACTIVATION_FUNC = "relu"
VALIDATION_PERCENT = 0.1
BATCH_SIZE = 64
NUM_CLASSES = -1
CHAR_TO_INT = {}
INT_TO_CHAR = {}

# =============================================================================
# FUNCTIONS START HERE
# =============================================================================
def reset_random_seeds():
    os.environ['PYTHONHASHSEED']=str(1)
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)

# =============================================================================
def save_metric_plot(model_name, model_type, l2_penalty, metric, trainX, validX):
    metric_prettify = ut.snake_case_prettify(metric)
    title = "{} for {}".format(metric_prettify, model_name)
    fig, ax = plt.subplots()
    epoch_axis = list(range(1, MAX_EPOCH_LENGTH + 1))
    ax.plot(epoch_axis, trainX, label='Training')
    ax.plot(epoch_axis, validX, label='Validation')
    ax.set_xlabel('Epochs')         # Add an x-label to the axes.
    ax.set_ylabel(metric_prettify)  # Add a y-label to the axes.
    ax.set_title(title)             # Add a title to the axes.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend();  # Add a legend.
    fig.savefig(f"{model_type}_{l2_penalty}_{metric}.png")

# =============================================================================
def generate_input_output_pairs(sequence):

    # Extract all input_output pairs from all sequences
    input_output_pairs = []
    for seq in sequence:
        for start in range(len(seq)-KMER_LENGTH):
            end = start + KMER_LENGTH
            seq_in = seq[start:end]
            seq_out = seq[end]
            input_output_pairs.append((seq_in, seq_out))

    return input_output_pairs

# =============================================================================
def preprocess_data(dataset):
    """
    Preprocesses raw dataset and returns tuple (dataX, dataY)
    """

    # First, convert the raw strings to integers
    input_as_lst = []
    output_as_lst = []
    for inp, out in dataset:
        input_as_lst.append([CHAR_TO_INT[c] for c in inp])
        output_as_lst.append(CHAR_TO_INT[out])

    # reshape X to be [samples, time steps, features], normalize
    dataX = np.reshape(input_as_lst, (len(input_as_lst), KMER_LENGTH, 1))
    dataX = (dataX - dataX.min()) / (dataX.max() - dataX.min())

    # Convert output to categorical vector
    dataY = np_utils.to_categorical(output_as_lst, num_classes=NUM_CLASSES)

    return dataX, dataY

# =============================================================================
def get_sequence_predictions(model, seq, gap_char):

    # Characters that already exist have a probability of 1. Until gaps are filled, their probability is 0
    predictions_probabilities = [(c, 1 if c != gap_char else 0) for c in seq]

    if not model:
        return predictions_probabilities

    for start in range(len(seq) - KMER_LENGTH):
        end = start+KMER_LENGTH
        # Only if we have a gap, do we need to update predictions_probabilities
        if seq[end] == gap_char:
            input_seq = [c for c, _ in predictions_probabilities[start:end]]
            input_seq = np.array([CHAR_TO_INT[c] for c in input_seq])
            input_seq = input_seq / float(NUM_CLASSES)
            input_seq = np.reshape(input_seq, (1, KMER_LENGTH, 1))

            output_arr = model.predict(input_seq).flatten()
            highest_probability = np.amax(output_arr)
            output_class = np.where(output_arr == highest_probability)[0][0]

            # Convert the output class integer back into the predicted character
            predicted_char = INT_TO_CHAR[output_class]
            predictions_probabilities[end] = (predicted_char, highest_probability)

    return predictions_probabilities

# =============================================================================
def predict_gaps(seq, forward_model=None, reverse_model=None, gap_char="-"):

    forward_preds = get_sequence_predictions(forward_model, seq, gap_char)
    reverse_preds = get_sequence_predictions(reverse_model, seq[::-1], gap_char)

    predicted_seq = ""
    for ((forward_pred, forward_prob), (reverse_pred, reverse_prob)) in zip(forward_preds, reverse_preds[::-1]):
        best_prediction = forward_pred if forward_prob >= reverse_prob else reverse_pred
        predicted_seq += best_prediction

    return predicted_seq

# =============================================================================
def build_lstm_model(l2_penalty):

    l2_regularizer = keras.regularizers.L2(l2_penalty)
    regularizer_kwargs = {
        "kernel_regularizer": l2_regularizer,
        # "bias_regularizer": l2_regularizer,
        # "activity_regularizer": l2_regularizer,
    }

    inputs = keras.Input(shape=(KMER_LENGTH, 1))
    outputs = keras.layers.LSTM(FIRST_LSTM_LAYER, **regularizer_kwargs)(inputs)
    outputs = keras.layers.Dense(FIRST_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = keras.layers.Dense(SECOND_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = keras.layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    return (keras.Model(inputs=inputs, outputs=outputs), "LSTM")

# ============================================================================
def build_cnn_lstm_model(l2_penalty):

    l2_regularizer = keras.regularizers.L2(l2_penalty)
    regularizer_kwargs = {
        "kernel_regularizer": l2_regularizer,
        # "bias_regularizer": l2_regularizer,
        # "activity_regularizer": l2_regularizer,
    }
    inputs = keras.Input(shape=(KMER_LENGTH, 1))
    outputs = keras.layers.Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE, **regularizer_kwargs)(inputs)
    outputs = keras.layers.LSTM(FIRST_LSTM_LAYER, **regularizer_kwargs)(outputs)
    outputs = keras.layers.Dense(FIRST_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = keras.layers.Dense(SECOND_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = keras.layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    return (keras.Model(inputs=inputs, outputs=outputs), "CNN LSTM")

# =============================================================================
def build_bilstm_model(l2_penalty):

    l2_regularizer = keras.regularizers.L2(l2_penalty)
    regularizer_kwargs = {
        "kernel_regularizer": l2_regularizer,
        # "bias_regularizer": l2_regularizer,
        # "activity_regularizer": l2_regularizer,
    }

    inputs = keras.Input(shape=(KMER_LENGTH, 1))
    outputs = keras.layers.Bidirectional(keras.layers.LSTM(FIRST_LSTM_LAYER, **regularizer_kwargs))(inputs)
    outputs = keras.layers.Dense(FIRST_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = keras.layers.Dense(SECOND_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = keras.layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    return (keras.Model(inputs=inputs, outputs=outputs), "Bi-LSTM")

# =============================================================================
def build_cnn_bilstm_model(l2_penalty):

    l2_regularizer = keras.regularizers.L2(l2_penalty)
    regularizer_kwargs = {
        "kernel_regularizer": l2_regularizer,
        # "bias_regularizer": l2_regularizer,
        # "activity_regularizer": l2_regularizer,
    }
    inputs = keras.Input(shape=(KMER_LENGTH, 1))
    outputs = keras.layers.Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE, **regularizer_kwargs)(inputs)
    outputs = keras.layers.Bidirectional(keras.layers.LSTM(FIRST_LSTM_LAYER, **regularizer_kwargs))(inputs)
    outputs = keras.layers.Dense(FIRST_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = keras.layers.Dense(SECOND_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = keras.layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    return (keras.Model(inputs=inputs, outputs=outputs), "CNN Bi-LSTM")

# =============================================================================
def get_train_valid_split(training_seqs):

    training_pairs = generate_input_output_pairs(training_seqs)

    # Shuffle the training data so no bias is introduced when splitting for validation
    np.random.shuffle(training_pairs)

    # Determine indices to use to split randomized data into training/validation/test sets
    validation_threshold = int(VALIDATION_PERCENT * len(training_pairs))

    # Convert lists of lists to appropriate data structure complete with any necessary preprocessing
    trainX, trainY = preprocess_data(training_pairs[validation_threshold:])
    validX, validY = preprocess_data(training_pairs[:validation_threshold])

    return trainX, trainY, validX, validY

# =============================================================================
def get_model_builder(model_type):
    model_types = {
        "lstm": build_lstm_model,
        "cnn_lstm": build_cnn_lstm_model,
        "bilstm": build_bilstm_model,
        "cnn_bilstm": build_cnn_bilstm_model,
    }
    if model_type not in model_types:
        raise Exception("Not a valid model type! Pick from {}".format(model_types.keys()))
    return model_types[model_type]

# =============================================================================
def compile_and_fit_model(model, trainX, trainY, validX, validY, early_stopping):
    model.compile(loss=COST_FUNC, optimizer=OPTIMIZER, metrics=['categorical_accuracy'])

    callbacks = []
    if early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=10))

    history = model.fit(
        trainX,
        trainY,
        epochs=MAX_EPOCH_LENGTH,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=(validX, validY),
        callbacks=callbacks
    )
    return history

# =============================================================================
def train_model(model_type, training_seqs, l2_penalty, early_stopping):
    reset_random_seeds()
    trainX, trainY, validX, validY = get_train_valid_split(training_seqs)
    model_builder = get_model_builder(model_type)
    model, model_type_name = model_builder(l2_penalty)
    history = compile_and_fit_model(model, trainX, trainY, validX, validY, early_stopping)

    return model, history, model_type_name

# =============================================================================
def main():

    reset_random_seeds()

    ut.mkdir_if_not_exists(RESULTS_PATH)

    cwd = os.getcwd()
    os.chdir(DATAPATH)

    sequences_to_train_on=100
    training_sequences = ut.get_sequences("training_sequences.txt")
    training_sequences = training_sequences[:sequences_to_train_on]
    training_sequences_reversed = [item[::-1] for item in training_sequences]

    de_novo_sequence = ut.get_sequences("de_novo_sequence.txt")[0]
    target_sequence = ut.get_sequences("target_sequence.txt")[0]
    target_sequence_reversed = target_sequence[::-1]

    # extract all chars from all sequences to create our mappings and to determine classes
    all_chars = set("".join(training_sequences) + target_sequence)

    # These globals must be determined at runtime
    global NUM_CLASSES, CHAR_TO_INT, INT_TO_CHAR
    NUM_CLASSES = len(all_chars)
    CHAR_TO_INT = {c: i for i, c in enumerate(all_chars)}
    INT_TO_CHAR = {v: k for k, v in CHAR_TO_INT.items()}

    # Only using cnn lstm for the current paper
    model_types = [
        ("lstm", [1.0E-8]),
        ("cnn_lstm", [1.0E-3]),
        ("bilstm", [1.0E-7]),
        ("cnn_bilstm", [1.0E-8])
    ]

    os.chdir(RESULTS_PATH)
    if not os.path.isfile("test_accuracies.csv"):
        headers = ["Model Type", "L2 Penalty", "Full Accuracy", "Gap Accuracy"]
        with open(f"test_accuracies.csv", "w+", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    for model_type, l2_penalties in model_types:
        for l2_penalty in l2_penalties:
            print("Training model_type: %s, l2_penalty: %s" % (model_type, l2_penalty))
            forward_model, forward_history, model_name = train_model(model_type, training_sequences, l2_penalty, early_stopping=False)
            reverse_model, _, _ = train_model(model_type, training_sequences_reversed, l2_penalty, early_stopping=False)

            # Forward model is trained on forward data, tested on forward data
            testing_pairs = generate_input_output_pairs([target_sequence])
            testX, testY = preprocess_data(testing_pairs)
            _, forward_accuracy = forward_model.evaluate(testX, testY)

            # Reverse model is trained on reverse data, tested on reverse data
            testing_pairs = generate_input_output_pairs([target_sequence_reversed])
            testX, testY = preprocess_data(testing_pairs)
            _, reverse_accuracy = reverse_model.evaluate(testX, testY)

            pred_sequence_full = predict_gaps(de_novo_sequence, forward_model, reverse_model)

            # FIXME: This is stupid, but can't be bothered to convert everything to numpy
            predicted_sequence_as_arr = np.array([c for c in pred_sequence_full])
            target_sequence_as_arr = np.array([c for c in target_sequence])
            de_novo_sequence_as_arr = np.array([c for c in de_novo_sequence])

            indices_to_predict = np.where(de_novo_sequence_as_arr != target_sequence_as_arr)[0]
            gap_indices = np.where(de_novo_sequence_as_arr == "-")[0]

            incorrect_indices = np.intersect1d(indices_to_predict, np.where(target_sequence_as_arr != predicted_sequence_as_arr)[0])
            correct_indices = np.intersect1d(indices_to_predict, np.where(target_sequence_as_arr == predicted_sequence_as_arr)[0])
            correct_gap_indices = np.intersect1d(gap_indices, correct_indices)

            gap_acc = len(correct_gap_indices) / len(gap_indices)
            full_acc = len(np.where(target_sequence_as_arr == predicted_sequence_as_arr)[0]) / len(target_sequence_as_arr)

            # Print to console
            print(f"Accuracy on Forward {model_name}, {l2_penalty}: {forward_accuracy:.2f}")
            print(f"Accuracy on Reverse {model_name}, {l2_penalty}: {reverse_accuracy:.2f}")
            print(f"Gap Acc for {model_name}, {l2_penalty}: {gap_acc}")
            print(f"Full Acc for {model_name}, {l2_penalty}: {full_acc}")
            ut.print_sequence(predicted_sequence_as_arr, f"Results on Original Scaffold with {model_name}, {l2_penalty}", incorrect_indices, correct_indices)

            # Save stuff to files
            with open(f"test_accuracies.csv", "a+", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([model_name, l2_penalty, full_acc, gap_acc])

            hist = forward_history.history
            train_acc = hist["categorical_accuracy"]
            valid_acc = hist["val_categorical_accuracy"]
            train_loss = hist["loss"]
            valid_loss = hist["val_loss"]

            headers = ["Training Accuracy", "Validation Accuracy", "Training Loss", "Validation Loss"]
            with open(f"{model_type}_{l2_penalty}_accuracy_and_loss.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(zip(train_acc, valid_acc, train_loss, valid_loss))

            ut.write_protein_scaffold_image(predicted_sequence_as_arr, incorrect_indices, correct_indices, f"{model_type}_predictions")

            save_metric_plot(model_name, model_type, l2_penalty, "loss", hist["loss"], hist["val_loss"])
            save_metric_plot(model_name, model_type, l2_penalty, "categorical_accuracy", hist["categorical_accuracy"], hist["val_categorical_accuracy"])

            parameters = {
                "kmer_length": KMER_LENGTH,
                "max_epoch_length": MAX_EPOCH_LENGTH,
                "first_dense_layer": FIRST_DENSE_LAYER,
                "second_dense_layer": SECOND_DENSE_LAYER,
                "num_filters": NUM_FILTERS,
                "filter_size": FILTER_SIZE,
                "first_lstm_layers": FIRST_LSTM_LAYER,
                "cost_func": COST_FUNC,
                "optimizer": OPTIMIZER,
                "output_activation_func": OUTPUT_ACTIVATION_FUNC,
                "hidden_layer_activation_func": HIDDEN_LAYER_ACTIVATION_FUNC,
                "validation_percent": VALIDATION_PERCENT,
                "num_classes": NUM_CLASSES,
                "batch_size": BATCH_SIZE,
                "l2_penalty": l2_penalty,
            }

            result = {
                "test_accuracy": full_acc,
                "val_categorical_accuracy": hist["val_categorical_accuracy"][-1],
                "categorical_accuracy": hist["categorical_accuracy"][-1],
                "loss": hist["loss"][-1],
                "val_loss": hist["val_loss"][-1],
                "epochs": MAX_EPOCH_LENGTH,
                "parameters": parameters,
            }

            path = os.path.join(RESULTS_PATH, f"{model_type}_results.json")
            with open(path, "w+") as f:
                f.write(json.dumps(result))

            path = os.path.join(RESULTS_PATH, f"{model_type}_model_summary.txt")
            with open(path, "w+") as f:
                forward_model.summary(print_fn=lambda x: f.write(x + '\n'))

    os.chdir(cwd)

if __name__ == "__main__":
    main()
