import os
os.system('color')
import random
from termcolor import colored
from typing import List, Optional
from copy import deepcopy
import csv
from enum import Enum

import tensorflow as tf
from keras import layers, Model
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator

import utility as ut

os.system('color')
random.seed(0)
np.random.seed(0)

DATAPATH = os.path.join(os.getcwd(), "data")
RESULTS_PATH = os.path.join(os.getcwd(), "autoencoder_results")

# =============================================================================
class Encoder(layers.Layer):
    """Encoder part of autoencoder"""

    # -------------------------------------------------------------------------
    def __init__(self, conv1_filters, conv2_filters, conv1_filter_size=5, conv2_filter_size=5, maxpool=2, dropout=0.25, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)

        self.conv1 = layers.Conv1D(conv1_filters, conv1_filter_size, padding="same", activation="relu")
        self.conv2 = layers.Conv1D(conv2_filters, conv2_filter_size, padding="same", activation="relu")
        self.maxpool = layers.MaxPooling1D(maxpool, padding="same")
        self.dropout = layers.Dropout(dropout)

    # -------------------------------------------------------------------------
    def call(self, inputs):
        return self.dropout(self.maxpool(self.conv2(self.dropout(self.maxpool(self.conv1(inputs))))))

# =============================================================================
class Decoder(layers.Layer):
    """Decoder part of autoencoder"""

    # -------------------------------------------------------------------------
    def __init__(self, conv1_filters, conv2_filters, conv1_filter_size=5, conv2_filter_size=5, maxpool=2, dropout=0.25, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)

        self.conv1 = layers.Conv1DTranspose(conv1_filters, conv1_filter_size, padding="same", activation="relu")
        self.conv2 = layers.Conv1DTranspose(conv2_filters, conv2_filter_size, padding="same", activation="relu")
        self.upsample = layers.UpSampling1D(maxpool)
        self.dropout = layers.Dropout(dropout)

    # -------------------------------------------------------------------------
    def call(self, inputs):
        return self.dropout(self.upsample(self.conv2(self.dropout(self.upsample(self.conv1(inputs))))))

# =============================================================================
class Autoencoder(Model):
    """Autoencoder"""
    # -------------------------------------------------------------------------
    def __init__(self, num_classes, name="autoencoder", **kwargs):
        super().__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.hyperparameters = {
            "conv1_filters": 32,
            "conv2_filters": 64,
            "conv1_filter_size": 5,
            "conv2_filter_size": 5,
            "bridge_filters": 128,
            "bridge_filter_size": 5,
            "dropout": 0.25,
            "maxpool": 2,
        }

        hp = self.hyperparameters

        self.encoder = Encoder(hp["conv1_filters"], hp["conv2_filters"])
        self.bridge = layers.Conv1D(hp["bridge_filters"], hp["bridge_filter_size"], padding="same", activation="relu")
        self.decoder = Decoder(hp["conv2_filters"], hp["conv1_filters"])
        self.finallayer = layers.Conv1D(self.num_classes, hp["conv1_filter_size"], padding="same", activation="softmax")

    # -------------------------------------------------------------------------
    def call(self, inputs):
        return self.finallayer(self.decoder(self.bridge(self.encoder(inputs))))

# =============================================================================
class PaddingType(Enum):
    ZERO=1
    TRUNCATE=2
    EMPTY=3

# =============================================================================
class NoisificationMethod(Enum):
    RANDOMSCATTER=1
    RANDOMCONTIG=2

# =============================================================================
class ProteinScaffoldFixer():
    """Class to correct errors in a protein scaffold and fill gaps"""

    # -------------------------------------------------------------------------
    def __init__(self, 
                output_seqs, 
                paddingtype=PaddingType.EMPTY, 
                noise_percent=0.2, 
                noisemethod=NoisificationMethod.RANDOMSCATTER, 
                numgaps=5, 
                mingapsize=3, 
                mincontigsize=1, 
                epochs=200,
                optimizer="adam",
                early_stopping=False):

        self.epochs = epochs
        self.optimizer = optimizer
        self.paddingtype = paddingtype
        self.noise_percent = noise_percent
        self.early_stopping = early_stopping

        self.classes = np.array(["-", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'])
        self.ohe = OneHotEncoder(sparse_output=False, categories=[self.classes])

        self.max_seq_length = max((len(seq) for seq in output_seqs))
        self.maxpool = 2
        while self.max_seq_length % (self.maxpool * self.maxpool) != 0:
            self.max_seq_length += 1

        self.output_seqs = output_seqs
        self.input_seqs = self.noisify_sequences(output_seqs, noise_percent, noisemethod, numgaps, mingapsize, mincontigsize)

        self.train_x = self.preprocess_sequences(self.input_seqs)
        self.train_y = self.preprocess_sequences(self.output_seqs)

        self.history = None

        self.autoencoder = Autoencoder(len(self.classes))

    # =============================================================================
    def noisify_sequences(self, seqs, noise_percent, noisemethod, numgaps=5, mingapsize=3, mincontigsize=3):
        if noisemethod == NoisificationMethod.RANDOMSCATTER:
            return self.noisify_by_random_scatter(seqs, noise_percent)
        elif noisemethod == NoisificationMethod.RANDOMCONTIG:
            return self.noisify_by_random_contigs(seqs, noise_percent, numgaps, mingapsize, mincontigsize)

    # =============================================================================
    def noisify_by_random_scatter(self, seqs, noise_percent):
        sequences = deepcopy(seqs)
        # To noisify our input data, we will replace random amino acids with something else
        for seq in sequences:

            # We randomly sample from all the possible indices of seq
            indices_to_replace = random.sample(range(len(seq)), int(noise_percent * len(seq)))
            seq[indices_to_replace] = "-"

        return sequences

    # =============================================================================
    def noisify_by_random_contigs(self, seqs, noise_percent, numgaps, mingapsize, mincontigsize):

        sequences = deepcopy(seqs)
        for seqind, seq in enumerate(sequences):

            amino_acid_length = len(seq)
            amino_acids_to_replace = int(amino_acid_length * noise_percent)

            # The idea is to build a gap queue, randomly putting each amino acid to replace into each gap
            # Then, we build a contig queue, randomly putting each amino acid into each contig
            gap_queue = np.zeros(numgaps).astype(int)
            for _ in range(amino_acids_to_replace):

                # Any gap can be considered
                valid_gaps = np.arange(numgaps)

                # ... so long as we have no underfilled gaps
                underfilled = gap_queue < mingapsize
                # If we have any underfilled, then consider only those until they are no longer underfilled
                if np.any(underfilled):
                    valid_gaps = valid_gaps[np.where(underfilled)]
                
                # Once valid gap indices have been determined, randomly pick one to increment
                gap_queue[np.random.choice(valid_gaps)] += 1

            # There can always be one more contig than gaps (if there's a contig at beginnning and end of sequence)
            contig_queue = np.zeros(numgaps+1).astype(int)

            # We have to allocate all the amino acids NOT in gaps into contigs BETWEEN the gaps
            for _ in range(amino_acid_length - amino_acids_to_replace):

                # Any contig can be considered
                valid_contigs = np.arange(len(contig_queue))

                # ... so long as we have no underfilled contigs
                underfilled = contig_queue < mincontigsize

                # The exceptions are the first and last contigs. They are never considered underfilled
                underfilled[0] = False
                underfilled[-1] = False

                # If we have any underfilled, then consider only those until they are no longer underfilled
                if np.any(underfilled):
                    valid_contigs = valid_contigs[np.where(underfilled)]
                
                # Once valid gap indices have been determined, randomly pick one to increment
                contig_queue[np.random.choice(valid_contigs)] += 1
            
            # Once we have determined gap_queue and contig_queue, we iterate over them to set the gaps equal
            # to our blank amino acid character

            sequence_pointer = 0
            iscontig=True
            while sequence_pointer < len(seq):
                if iscontig:
                    # Don't do anything for the contig except increment the pointer and pop off the contig queue
                    sequence_pointer += contig_queue[0]
                    contig_queue = np.delete(contig_queue, 0)
                else:
                    seq[sequence_pointer:sequence_pointer+gap_queue[0]] = '-'
                    sequence_pointer += gap_queue[0]
                    gap_queue = np.delete(gap_queue, 0)

                # We alternate between contigs and gaps
                iscontig = not iscontig

        return sequences

    # -------------------------------------------------------------------------
    def predict_sequence(self, seq, predict_only_gaps):

        scaffold = self.preprocess_sequences([seq])
        pred = self.autoencoder.predict(scaffold).reshape(self.max_seq_length, len(self.classes))

        # Set the probability of empty "-" to zero, since we always want to predict something
        emptyclass = np.where(self.ohe.transform(np.array("-").reshape(-1, 1))[0])[0][0]
        pred[:, emptyclass] = 0.0

        # Convert the probability distribution to a one-hot encoded vector
        mask = pred == np.amax(pred, axis=1).reshape(pred.shape[0], 1)
        indices = list((i, np.where(mask[i])[0][0]) for i in range(mask.shape[0]))

        pred = np.zeros(pred.shape)
        for i in indices:
            pred[i] = 1

        # Then we can use our one hot encoder to convert back to the original sequence of classes
        pred = self.ohe.inverse_transform(pred[:len(seq), :len(self.classes)]).reshape(len(seq))

        if predict_only_gaps:
            # We only care about predicting the gaps in seq, so replace amino acids in prediction with original nongaps
            nongaps = np.where(seq != '-')[0]
            pred[nongaps] = seq[nongaps]

        return pred

    # =============================================================================
    def preprocess_sequences(self, seqs: List[np.array]) -> np.array:

        # the value -1 lets numpy know to infer the shape. So it's just a column vector of length num_samples
        seqs = [np.array(seq).reshape(-1, 1) for seq in seqs]

        # One-hot encode each sequence
        seqs = [self.ohe.fit_transform(seq) for seq in seqs]

        if self.paddingtype == PaddingType.ZERO:
            # The sequences may have different lengths, so we will pad them with zeros
            # We are padding to fill up to max length, then we can turn it into a single numpy tensor
            return np.array([np.pad(seq, ((0, self.max_seq_length-len(seq)), (0, 0))) for seq in seqs])

        elif self.paddingtype == PaddingType.EMPTY:

            # We pad with the "empty" class
            emptyclass = self.ohe.fit_transform(np.array("-").reshape(-1, 1))[0]

            # dynamically extend each seq by enough emptyvals to make a single sequence length
            return np.array([np.vstack((seq, *(emptyclass for _ in range(self.max_seq_length-len(seq))))) for seq in seqs])

        elif self.paddingtype == PaddingType.TRUNCATE:
            # FIXME: Implement me
            raise Exception("Have not implemented Padding Type TRUNCATE!")

    # =============================================================================
    def train(self, verbose="auto", optimizer="adam", epochs=None):
        self.optimizer = optimizer
        if epochs:
            self.epochs = epochs

        self.autoencoder.compile(optimizer=self.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        callbacks = []
        if self.early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10))
        self.history = self.autoencoder.fit(self.train_x, self.train_y, epochs=self.epochs, validation_split=0.15, verbose=verbose, callbacks=callbacks)

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
def get_sequences(fasta_file: str) -> List[np.array]:
    sequences = []
    lines = []
    with open(fasta_file, "r") as input_file:
        lines = list(filter(None, input_file.read().split("\n")))

    parts = []
    for line in lines:
        if line.startswith(">"):
            if parts:
                sequences.append(np.array([c for c in "".join(parts)]))
            parts = []
        else:
            parts.append(line)
    if parts:
        sequences.append(np.array([c for c in "".join(parts)]))
    return sequences

# =============================================================================
def snake_case_prettify(s):
    return " ".join(w.capitalize() for w in s.split("_"))

# =============================================================================
def save_models(models):
    modeldir = os.path.join(os.getcwd(), "1Dmodels")
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    for name, model in models.items():
        model.save(os.path.join(modeldir, name))

# =============================================================================
def load_models():
    modeldir = os.path.join(os.getcwd(), "1Dmodels")
    if os.path.exists(modeldir):
        return {f: tf.keras.models.load_model(os.path.join(modeldir, f)) for f in os.listdir(modeldir)}
    return {}

# =============================================================================
def print_diff_between_target_and_de_novo():

    dataset_dir = os.path.join(os.getcwd(), "newdatasets")
    
    de_novo_sequence = get_sequences(os.path.join(dataset_dir, "denovo_0.20_6.txt"))[0]
    target_sequence = get_sequences(os.path.join(dataset_dir, "target_sequence.txt"))[0]

    gap_indices = np.where(de_novo_sequence == '-')[0]
    incorrect_indices = np.where(de_novo_sequence != target_sequence)[0]
    correct_indices = np.where(de_novo_sequence == target_sequence)[0]
    incorrect_non_gaps = np.setdiff1d(incorrect_indices, gap_indices)
    print(f"Length of target: {len(target_sequence)}")
    print(f"Number of incorrect non-gaps: {len(incorrect_non_gaps)}")
    print(f"Number of gaps: {len(gap_indices)}")

    print_sequence(de_novo_sequence, "Protein Scaffold", incorrect_indices, correct_indices)

# =============================================================================
def test_original_protein_scaffold():

    ut.mkdir_if_not_exists(RESULTS_PATH)
    ut.mkdir_if_not_exists(DATAPATH)

    epochs = 200
    noise_percent = 0.25

    cwd = os.getcwd()

    # change to the location of the data for reading
    os.chdir(DATAPATH)
    de_novo_sequence = get_sequences("de_novo_sequence.txt")[0]
    target_sequence = get_sequences("target_sequence.txt")[0]
    alltrainingdata = get_sequences("training_sequences.txt")
    metadata = "training_sequences_metadata.csv"

    # then change to the results directory for writing
    os.chdir(RESULTS_PATH)

    sequences_to_train_on = [1000]
    # sequences_to_train_on = [100, 500, 1000, 2000]

    trainfilename = "cda_breakdown_breakdown_of_numtrain.csv"
    headers = ["Num Training Instances", "Full Accuracy", "Gap Accuracy", "Nongap Accuracy"]
    with open(trainfilename, "w+", encoding="utf-8", newline="") as f:
        train_writer = csv.writer(f)
        train_writer.writerow(headers)

    train_file = open(trainfilename, "a+", encoding="utf-8", newline="")
    train_writer = csv.writer(train_file)

    for numtrain in sequences_to_train_on:

        trainingdata = alltrainingdata[:numtrain]
        random.shuffle(trainingdata)

        incorrect_indices = np.where(de_novo_sequence != target_sequence)[0]
        correct_indices = np.where(de_novo_sequence == target_sequence)[0]

        fixer = ProteinScaffoldFixer(trainingdata, noise_percent=noise_percent, epochs=epochs)
        fixer.train(verbose=1)

        hist = fixer.history.history
        training_loss = hist["loss"]
        validation_loss = hist["val_loss"]
        training_acc = hist["accuracy"]
        validation_acc = hist["val_accuracy"]

        headers = ["Training Accuracy", "Validation Accuracy", "Training Loss", "Validation Loss"]
        with open(f"autoencoder_results_on_original_{numtrain}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(zip(training_acc, validation_acc, training_loss, validation_loss))

        predict_only_gaps = False
        pred = fixer.predict_sequence(de_novo_sequence, predict_only_gaps)

        incorrect_indices = np.where(target_sequence != pred)[0]
        correct_indices = np.where(target_sequence == pred)[0]
        indices_to_predict = np.where(de_novo_sequence != target_sequence)[0]
        correct_indices_to_predict = np.intersect1d(correct_indices, indices_to_predict)

        gap_indices = np.where(de_novo_sequence == "-")[0]
        error_indices = np.setdiff1d(indices_to_predict, gap_indices)

        correct_gap_indices = np.intersect1d(gap_indices, correct_indices)
        correct_error_indices = np.intersect1d(error_indices, correct_indices)

        gap_acc = len(correct_gap_indices) / len(gap_indices)
        err_acc = len(correct_error_indices) / len(error_indices)
        full_acc = len(np.where(target_sequence == pred)[0]) / len(target_sequence)

        # print to console the prediction
        print_sequence(pred, f"Predicted for {numtrain}", incorrect_indices, correct_indices_to_predict)

        # write the final accuracies according to the number of training instances
        with open(trainfilename, "a+", encoding="utf-8", newline="") as f:
            train_writer = csv.writer(f)
            train_writer.writerow([numtrain, full_acc, gap_acc, err_acc])

        # and save the predictions as images for use in the paper
        ut.write_protein_scaffold_image(pred, incorrect_indices, correct_indices_to_predict, f"cda_predictions_on_original_{numtrain}")

    train_file.close()
    os.chdir(cwd)

# =============================================================================
def test_generated_datasets():

    ut.mkdir_if_not_exists(RESULTS_PATH)
    ut.mkdir_if_not_exists(DATAPATH)

    epochs = 200
    noise_percent = 0.25

    cwd = os.getcwd()

    os.chdir(DATAPATH)
    target_sequence = get_sequences("target_sequence.txt")[0]

    sequences_to_train_on = np.array([1000])
    # sequences_to_train_on = np.array([100, 500, 1000, 2000])
    percent_missings = np.array([0.20, 0.30, 0.40])
    num_gaps = np.array([4, 6, 8, 10])

    resultsfilename = "cda_generated_accuracies.csv"

    os.chdir(RESULTS_PATH)
    with open(resultsfilename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Percent Missing", "Num Gaps", "Training Instances", "Full Accuracy", "Gap Accuracy", "Non-gap Accuracy", "Most Similar Reference", "Least Similar Reference"])

    for percent_missing in percent_missings:
        for num_gap in num_gaps:

            os.chdir(DATAPATH)

            scaffold_sequence = get_sequences(f"denovo_{percent_missing:.2f}_{num_gap}.txt")[0]
            alltrainingdata = get_sequences(f"training_{percent_missing:.2f}_{num_gap}.txt")
            metadata = []

            with open(f"training_{percent_missing:.2f}_{num_gap}_metadata.csv", "r") as f:
                metadata = list(csv.reader(f))

            for num_train in sequences_to_train_on:

                most_similar = metadata[0][2]
                least_similar = metadata[num_train-1][2]

                trainingdata = alltrainingdata[:num_train]
                fixer = ProteinScaffoldFixer(trainingdata, noise_percent=noise_percent, epochs=epochs, early_stopping=True)
                fixer.train(verbose=1)

                predict_only_gaps = False
                pred = fixer.predict_sequence(scaffold_sequence, predict_only_gaps)

                incorrect_indices = np.where(target_sequence != pred)[0]
                correct_indices = np.where(target_sequence == pred)[0]

                indices_to_predict = np.where(scaffold_sequence != target_sequence)[0]
                correct_indices_to_predict = np.intersect1d(correct_indices, indices_to_predict)

                gap_indices = np.where(scaffold_sequence == "-")[0]
                error_indices = np.setdiff1d(indices_to_predict, gap_indices)

                correct_gap_indices = np.intersect1d(gap_indices, correct_indices)
                correct_error_indices = np.intersect1d(error_indices, correct_indices)

                gap_acc = len(correct_gap_indices) / len(gap_indices)
                err_acc = len(correct_error_indices) / len(error_indices)
                full_acc = len(np.where(target_sequence == pred)[0]) / len(target_sequence)

                # Print to console
                print(f"Gap Acc for CDA: {gap_acc}")
                print(f"Err Acc for CDA: {err_acc}")
                print(f"Full Acc for CDA: {full_acc}")
                print_sequence(pred, f"Results on gap={num_gap}, {percent_missing}, {num_train}", incorrect_indices, correct_indices_to_predict)

                # And save results 
                os.chdir(RESULTS_PATH)
                with open(resultsfilename, "a+", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([percent_missing, num_gap, num_train, full_acc, gap_acc, err_acc, most_similar, least_similar])

                # Also save the images for the paper
                ut.write_protein_scaffold_image(pred, incorrect_indices, correct_indices_to_predict, f"de_novo_{percent_missing:.2f}_{num_gap}_{num_train}")

    os.chdir(cwd)

# =============================================================================
def main():

    # RUN THIS TO PRINT DIFFERENCE BETWEEN DE NOVO AND TARGET TO ENSURE ITS CORRECCT
    # print_diff_between_target_and_de_novo()

    # RUN THIS TO TRAIN A MODEL, SHOW PREDICTIONS ON ORIGINAL PROTEIN SCAFFOLD, GENERATE RESULTS
    test_original_protein_scaffold()

    # RUN THIS TO TRAIN ONE MODEL FOR EACH PROTEIN SCAFFOLD IN NEW DATASETS AND COLLECT RESULTS
    test_generated_datasets()

if __name__ == '__main__':
    main()
