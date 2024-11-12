
import functools
import json
import os
from termcolor import colored
from typing import Optional
from copy import deepcopy
import csv

import os
os.system('color')

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.utils import np_utils
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import utility as ut

# Use monospace to make formatting cleaner
plt.rcParams.update({'font.family':'monospace'})

# fix random seed for reproducibility
np.random.seed(7)

# =============================================================================
# Globals here
# =============================================================================

RESULTS_PATH = os.path.join(os.getcwd(), "results")
DATAPATH = os.path.join(os.getcwd(), "newdatasets")
LSTM_RESULTS_PATH = os.path.join(os.getcwd(), "lstm_results")

# Run 38 for bilstm was a really good run
# Run 1, 8 are decent for lstm

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

# =============================================================================
def snake_case_prettify(s):
    return " ".join(w.capitalize() for w in s.split("_"))

# =============================================================================
def get_run_number() -> int:
    run_number_file = os.path.join(RESULTS_PATH, "run_number.txt")
    if not os.path.exists(run_number_file):
        with open(run_number_file, "w+") as f:
            f.write(json.dumps({"run_number": 1}))

    with open(run_number_file, "r") as f:
        return json.loads(f.read())["run_number"]

# =============================================================================
def update_run_rumber():
    run_number_file = os.path.join(RESULTS_PATH, "run_number.txt")
    with open(run_number_file, "r") as f:
        run_number = int(json.loads(f.read())["run_number"]) + 1

    with open(run_number_file, "w+") as f:
        f.write(json.dumps({"run_number": run_number}))

# =============================================================================
# Save what we want from the results
def save_result(basedir, model_type, model_name, test_accuracy, history, model, l2_penalty):

    model_type_dir = os.path.join(basedir, model_type)
    l2_dir = os.path.join(model_type_dir, str(l2_penalty))
    ut.mkdir_if_not_exists(basedir)
    ut.mkdir_if_not_exists(model_type_dir)
    ut.mkdir_if_not_exists(l2_dir)

    save_metric_plot(l2_dir, model_name, "loss", history["loss"], history["val_loss"])
    save_metric_plot(l2_dir, model_name, "categorical_accuracy", history["categorical_accuracy"], history["val_categorical_accuracy"])

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
        "test_accuracy": test_accuracy,
        "val_categorical_accuracy": history["val_categorical_accuracy"][-1],
        "categorical_accuracy": history["categorical_accuracy"][-1],
        "loss": history["loss"][-1],
        "val_loss": history["val_loss"][-1],
        "epochs": MAX_EPOCH_LENGTH,
        "parameters": parameters,
    }

    path = os.path.join(RESULTS_PATH, l2_dir, "results.json")
    with open(path, "w+") as f:
        f.write(json.dumps(result))

    path = os.path.join(RESULTS_PATH, l2_dir, "model_summary.txt")
    with open(path, "w+") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

# =============================================================================
def save_metric_plot(filedir, model_name, metric, trainX, validX):
    metric_prettify = snake_case_prettify(metric)
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
    fig.savefig(os.path.join(RESULTS_PATH, filedir, f"{metric}.png"))

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
def get_nonmatching_indices(seq1, seq2):
    s = set()
    for i, (c1, c2) in enumerate(zip(seq1, seq2)):
        if c1 != c2:
            s.add(i)
    return s

# =============================================================================
def highlight_indices(seq: np.array, indices: np.array, color: str):
    # We use deepcopy to prevent mutation and we cast to object so that we can treat contents as python strings
    # otherwise, it gets messed up as it treats each element as a single character
    newseq = deepcopy(seq).astype('object')
    newseq[indices] = np.vectorize(lambda x: colored(x, color, attrs=["bold"]))(seq[indices])
    return newseq

# =============================================================================
def print_sequence(seq, 
                   header: str=None, 
                   incorrect_indices: Optional[np.array]=None, 
                   correct_indices: Optional[np.array]=None):

    newseq = deepcopy(seq)
    if correct_indices is not None and correct_indices.size != 0:
        newseq = highlight_indices(newseq, correct_indices, "green")
    if incorrect_indices is not None and incorrect_indices.size != 0:
        newseq = highlight_indices(newseq, incorrect_indices, "red")

    line_length = 40
    if header:
        print(header)
    print("=" * line_length)

    i = 0
    while i < len(newseq):
        print(" ".join(newseq[i: i+line_length]))
        i += line_length

# =============================================================================
def get_sequences(fasta_file):
    sequences = []
    lines = []
    with open(fasta_file, "r") as input_file:
        lines = list(filter(None, input_file.read().split("\n")))

    parts = []
    for line in lines:
        if line.startswith(">"):
            if parts:
                sequences.append("".join(parts))
            parts = []
        else:
            parts.append(line)
    if parts:
        sequences.append("".join(parts))
    return sequences

# =============================================================================
def build_lstm_model(l2_penalty):

    l2_regularizer = regularizers.L2(l2_penalty)
    regularizer_kwargs = {
        "kernel_regularizer": l2_regularizer,
        # "bias_regularizer": l2_regularizer,
        # "activity_regularizer": l2_regularizer,
    }

    inputs = keras.Input(shape=(KMER_LENGTH, 1))
    outputs = layers.LSTM(FIRST_LSTM_LAYER, **regularizer_kwargs)(inputs)
    outputs = layers.Dense(FIRST_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(SECOND_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    return (keras.Model(inputs=inputs, outputs=outputs), "LSTM")

# ============================================================================
def build_cnn_lstm_model(l2_penalty):

    l2_regularizer = regularizers.L2(l2_penalty)
    regularizer_kwargs = {
        "kernel_regularizer": l2_regularizer,
        # "bias_regularizer": l2_regularizer,
        # "activity_regularizer": l2_regularizer,
    }
    inputs = keras.Input(shape=(KMER_LENGTH, 1))
    outputs = layers.Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE, **regularizer_kwargs)(inputs)
    outputs = layers.LSTM(FIRST_LSTM_LAYER, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(FIRST_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(SECOND_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    return (keras.Model(inputs=inputs, outputs=outputs), "CNN LSTM")

# =============================================================================
def build_bilstm_model(l2_penalty):

    l2_regularizer = regularizers.L2(l2_penalty)
    regularizer_kwargs = {
        "kernel_regularizer": l2_regularizer,
        # "bias_regularizer": l2_regularizer,
        # "activity_regularizer": l2_regularizer,
    }

    inputs = keras.Input(shape=(KMER_LENGTH, 1))
    outputs = layers.Bidirectional(layers.LSTM(FIRST_LSTM_LAYER, **regularizer_kwargs))(inputs)
    outputs = layers.Dense(FIRST_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(SECOND_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    return (keras.Model(inputs=inputs, outputs=outputs), "Bi-LSTM")

# =============================================================================
def build_cnn_bilstm_model(l2_penalty):

    l2_regularizer = regularizers.L2(l2_penalty)
    regularizer_kwargs = {
        "kernel_regularizer": l2_regularizer,
        # "bias_regularizer": l2_regularizer,
        # "activity_regularizer": l2_regularizer,
    }
    inputs = keras.Input(shape=(KMER_LENGTH, 1))
    outputs = layers.Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE, **regularizer_kwargs)(inputs)
    outputs = layers.Bidirectional(layers.LSTM(FIRST_LSTM_LAYER, **regularizer_kwargs))(inputs)
    outputs = layers.Dense(FIRST_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(SECOND_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
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
def tune_l2_penalty(filedir, model_type, training_seqs):

    trainX, trainY, validX, validY = get_train_valid_split(training_seqs)
    model_builder = get_model_builder(model_type)

    l2_possibilities = np.array([
        [0.0, 0.0],
        [1.0E-5, 0.0],
        [1.0E-4, 0.0],
        [1.0E-3, 0.0],
        [1.0E-2, 0.0],
        [1.0E-1, 0.0]
    ])
    for row in l2_possibilities:

        model, model_type_name = model_builder(row[0])
        history = compile_and_fit_model(model, trainX, trainY, validX, validY, False)
        row[1] = history.history["val_loss"][-1]

    title = f"Loss versus Lambda for {model_type_name}"
    fig, ax = plt.subplots()

    ax.plot(l2_possibilities[:, 0], l2_possibilities[:, 1], label='L2 Vals')
    ax.set_xlabel('L2 Values')
    ax.set_ylabel('Loss')
    ax.set_title(title)

    ax.legend();  # Add a legend.
    fig.savefig(os.path.join(filedir, "lambda_curve.png"))

    best_lambda, smallest_loss = l2_possibilities[l2_possibilities[:, 0] == np.min(l2_possibilities[:, 0])][0]
    result = {"best_lambda": best_lambda, "smallest_loss": smallest_loss}
    with open(os.path.join(filedir, f"best_lambda.txt"), "w+") as f:
        return f.write(json.dumps(result))

# =============================================================================
def train_model(model_type, training_seqs, l2_penalty, early_stopping):
    trainX, trainY, validX, validY = get_train_valid_split(training_seqs)
    model_builder = get_model_builder(model_type)
    model, model_type_name = model_builder(l2_penalty)
    history = compile_and_fit_model(model, trainX, trainY, validX, validY, early_stopping)

    return model, history, model_type_name

# =============================================================================
def test_original_protein_scaffold():

    ut.mkdir_if_not_exists(LSTM_RESULTS_PATH)
    ut.mkdir_if_not_exists(DATAPATH)

    cwd = os.getcwd()
    os.chdir(DATAPATH)

    sequences_to_train_on=100
    training_sequences = get_sequences("training_sequences.txt")
    training_sequences = training_sequences[:sequences_to_train_on]
    training_sequences_reversed = [item[::-1] for item in training_sequences]

    de_novo_sequence = get_sequences("de_novo_sequence.txt")[0]
    target_sequence = get_sequences("target_sequence.txt")[0]
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
        # ("lstm", [1.0E-8]),
        ("cnn_lstm", [1.0E-3]),
        # ("bilstm", [1.0E-7]),
        # ("cnn_bilstm", [1.0E-8])
    ]

    os.chdir(LSTM_RESULTS_PATH)
    for model_type, l2_penalties in model_types:
        for l2_penalty in l2_penalties:
            print("Training model_type: %s, l2_penalty: %s" % (model_type, l2_penalty))
            forward_model, forward_history, model_name = train_model(model_type, training_sequences, l2_penalty, early_stopping=False)
            reverse_model, _, _ = train_model(model_type, training_sequences_reversed, l2_penalty, early_stopping=False)

            # Forward model is trained on forward data, tested on forward data
            testing_pairs = generate_input_output_pairs([target_sequence])
            testX, testY = preprocess_data(testing_pairs)
            # _, forward_accuracy = forward_model.evaluate(testX, testY)
            # print(f"Accuracy on Forward Model: {forward_accuracy:.2f}")

            # Reverse model is trained on reverse data, tested on reverse data
            testing_pairs = generate_input_output_pairs([target_sequence_reversed])
            testX, testY = preprocess_data(testing_pairs)
            # _, reverse_accuracy = reverse_model.evaluate(testX, testY)
            # print(f"Accuracy on Reverse Model: {reverse_accuracy:.2f}")

            hist = forward_history.history
            train_acc = hist["categorical_accuracy"]
            valid_acc = hist["val_categorical_accuracy"]
            train_loss = hist["loss"]
            valid_loss = hist["val_loss"]

            headers = ["Training Accuracy", "Validation Accuracy", "Training Loss", "Validation Loss"]
            with open(f"{model_type}_results_on_original.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(zip(train_acc, valid_acc, train_loss, valid_loss))

            pred_sequence_full = predict_gaps(de_novo_sequence, forward_model, reverse_model)

            # FIXME: This is stupid, but can't be bothered to convert everything to numpy
            predicted_sequence_as_arr = np.array([c for c in pred_sequence_full])
            target_sequence_as_arr = np.array([c for c in target_sequence])
            de_novo_sequence_as_arr = np.array([c for c in de_novo_sequence])

            indices_to_predict = np.where(de_novo_sequence_as_arr != target_sequence_as_arr)[0]
            gap_indices = np.where(de_novo_sequence_as_arr == "-")[0]
            error_indices = np.setdiff1d(indices_to_predict, gap_indices)

            incorrect_indices = np.intersect1d(indices_to_predict, np.where(target_sequence_as_arr != predicted_sequence_as_arr)[0])
            correct_indices = np.intersect1d(indices_to_predict, np.where(target_sequence_as_arr == predicted_sequence_as_arr)[0])
            correct_gap_indices = np.intersect1d(gap_indices, correct_indices)
            correct_error_indices = np.intersect1d(error_indices, correct_indices)

            gap_acc = len(correct_gap_indices) / len(gap_indices)
            err_acc = len(correct_error_indices) / len(error_indices)
            full_acc = len(np.where(target_sequence_as_arr == predicted_sequence_as_arr)[0]) / len(target_sequence_as_arr)

            print(f"Gap Acc for {model_type}: {gap_acc}")
            print(f"Err Acc for {model_type}: {err_acc}")
            print(f"Full Acc for {model_type}: {full_acc}")

            headers = ["Full Accuracy", "Gap Accuracy", "Non-gap Accuracy"]
            with open(f"{model_type}_accuracies_on_original.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerow([full_acc, gap_acc, err_acc])

            print_sequence(predicted_sequence_as_arr, f"Results on Original Scaffold with {model_type}", incorrect_indices, correct_indices)
            ut.write_protein_scaffold_image(predicted_sequence_as_arr, incorrect_indices, correct_indices, f"{model_type}_predictions_on_original")

    os.chdir(cwd)

# =============================================================================
def test_generated_datasets():

    ut.mkdir_if_not_exists(LSTM_RESULTS_PATH)
    ut.mkdir_if_not_exists(DATAPATH)

    cwd = os.getcwd()
    os.chdir(DATAPATH)

    target_sequence = get_sequences("target_sequence.txt")[0]
    target_sequence_reversed = target_sequence[::-1]

    # extract all chars from all sequences to create our mappings and to determine classes
    all_chars = "-ABCDEFGHIKLMNPQRSTVWXYZ"

    # These globals must be determined at runtime
    global NUM_CLASSES, CHAR_TO_INT, INT_TO_CHAR
    NUM_CLASSES = len(all_chars)
    CHAR_TO_INT = {c: i for i, c in enumerate(all_chars)}
    INT_TO_CHAR = {v: k for k, v in CHAR_TO_INT.items()}

    os.chdir(LSTM_RESULTS_PATH)
    with open("cnn_lstm_generated_datasets_results.csv", "w+", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Error Percent", "Contiguous Gaps", "Full Accuracy", "Gap Accuracy", "Error Accuracy", "Train Accuracy", "Validation Accuracy", "Train Loss", "Validation Loss"])

    runs = [
        (0.20, (4, 6, 8, 10)),
        (0.30, (4, 6, 8, 10)),
        (0.40, (4, 6, 8, 10)),
    ]

    sequences_to_train_on=100
    for percent_missing, numgaps in runs:
        for num_gap in numgaps:

            os.chdir(DATAPATH)

            de_novo_sequence = get_sequences(f"denovo_{percent_missing:.2f}_{num_gap}.txt")[0]
            trainingdata = get_sequences(f"training_{percent_missing:.2f}_{num_gap}.txt")[:sequences_to_train_on]
            trainingdata_reversed = [item[::-1] for item in trainingdata]

            forward_model, _, _ = train_model("cnn_lstm", trainingdata, 1.0E-3, early_stopping=True)
            reverse_model, _, _ = train_model("cnn_lstm", trainingdata_reversed, 1.0E-3, early_stopping=True)

            hist = forward_model.history.history
            train_acc = hist["categorical_accuracy"][-1]
            valid_acc = hist["val_categorical_accuracy"][-1]
            train_loss = hist["loss"][-1]
            valid_loss = hist["val_loss"][-1]

            pred_sequence_full = predict_gaps(de_novo_sequence, forward_model, reverse_model)

            predicted_sequence_as_arr = np.array([c for c in pred_sequence_full])
            target_sequence_as_arr = np.array([c for c in target_sequence])
            de_novo_sequence_as_arr = np.array([c for c in de_novo_sequence])

            indices_to_predict = np.where(de_novo_sequence_as_arr != target_sequence_as_arr)[0]
            gap_indices = np.where(np.array([c for c in de_novo_sequence]) == "-")[0]
            error_indices = np.setdiff1d(indices_to_predict, gap_indices)

            incorrect_indices = np.intersect1d(indices_to_predict, np.where(target_sequence_as_arr != predicted_sequence_as_arr)[0])
            correct_indices = np.intersect1d(indices_to_predict, np.where(target_sequence_as_arr == predicted_sequence_as_arr)[0])
            correct_gap_indices = np.intersect1d(gap_indices, correct_indices)
            correct_error_indices = np.intersect1d(error_indices, correct_indices)

            gap_acc = len(correct_gap_indices) / len(gap_indices)
            err_acc = len(correct_error_indices) / len(error_indices)
            full_acc = len(np.where(target_sequence_as_arr == predicted_sequence_as_arr)[0]) / len(target_sequence_as_arr)

            print(f"Gap Acc for CDA: {gap_acc}")
            print(f"Err Acc for CDA: {err_acc}")
            print(f"Full Acc for CDA: {full_acc}")

            os.chdir(LSTM_RESULTS_PATH)
            print_sequence(predicted_sequence_as_arr, f"CNN-LSTM Results on gap={num_gap}, {percent_missing}", incorrect_indices, correct_indices)
            with open("cnn_lstm_generated_datasets_results.csv", "a+", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([percent_missing, num_gap, full_acc, gap_acc, err_acc, train_acc, valid_acc, train_loss, valid_loss])
            ut.write_protein_scaffold_image(predicted_sequence_as_arr, incorrect_indices, correct_indices, f"denovo_{percent_missing:.2f}_{num_gap}_{sequences_to_train_on}")

    os.chdir(cwd)

# =============================================================================
def main():

    # RUN THIS TO TRAIN ALL FOUR MODELS, SHOW PREDICTIONS ON ORIGINAL PROTEIN SCAFFOLD, GENERATE RESULTS
    test_original_protein_scaffold()

    # RUN THIS TO TRAIN ONE MODEL FOR EACH PROTEIN SCAFFOLD IN NEW DATASETS AND COLLECT RESULTS
    test_generated_datasets()

if __name__ == "__main__":
    main()
