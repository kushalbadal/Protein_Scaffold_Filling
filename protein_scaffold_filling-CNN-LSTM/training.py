import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import  joblib

# Function to convert sequences to numeric representation
def sequence_to_numeric(sequence):
    return [CHAR_TO_INT[char] for char in sequence]

# Function to convert numeric representation back to sequence
def numeric_to_sequence(numeric_seq):
    return "".join([INT_TO_CHAR[num] for num in numeric_seq])


def remove_duplicate(word_list):
    unique_words = set()
    result = []

    for word in word_list:
        if word not in unique_words:
            unique_words.add(word)
            result.append(word)

    return result

def get_sequences(file_name):
    sequences = []
    lines = []
    with open(file_name, "r") as input_file:
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

def process_data(sequences):
    new_seq = []
    for amino in sequences:
        for i in range(len(amino) - 11 + 1):
            kmer = amino[i:i + 11]
            new_seq.append(kmer)
    new_seq = remove_duplicate(new_seq)
    dict = {}
    dict_for_prediction = {}
    for count, i in enumerate(new_seq, start=1):
        for j in range(0, len(i)):
            temp = i.replace(i[j], "-", 1)
            dict[temp] = count
        dict[i] = count
        dict_for_prediction[count] = i
    return dict ,dict_for_prediction

# Load training sequences
training_sequences = get_sequences("data/training_sequences.txt")
sequences_to_train_on = len(training_sequences)


# Load de novo sequence and its reverse
# de_novo_sequence = get_sequences("data/de_novo_sequence.txt")[0]
# print(de_novo_sequence)
de_novo_sequence = "---MTQSPSSLSASVGDRVTITCK---NIDKYLNWYQQKPGKAPKLLIYNTNNLQTGVPSRF---G----FTFTI-----------YCLQHISRPRTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFN----"

# de_novo_sequence = "--MTQSPS"

all_chars = set("".join(training_sequences) + de_novo_sequence)
NUM_CLASSES = len(all_chars)
CHAR_TO_INT = {c: i for i, c in enumerate(all_chars, start=1)}
INT_TO_CHAR = {v: k for k, v in CHAR_TO_INT.items()}


# Convert training sequences to numeric representation

X_train_data = []
y_train_data = []
training_seq_dict, pred_dict = process_data(training_sequences)
for keys, values in training_seq_dict.items():
    X_train_data = X_train_data + [sequence_to_numeric(keys)]
    y_train_data = y_train_data + [values]

# training_sequences_numeric = [sequence_to_numeric(seq) for seq in training_sequences]
max_seq_length = max(len(seq) for seq in X_train_data)

# Pad training sequences to the same length
padded_training_sequences_numeric = [seq + [0] * (max_seq_length - len(seq)) for seq in X_train_data]

X_train = np.array(padded_training_sequences_numeric)
y_train = np.array(y_train_data)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Train the KNN classifier for forward sequences
k = 3
knn_classifier_forward = KNeighborsClassifier(n_neighbors=k)

# Use tqdm to visualize training progress
with tqdm(total=len(X_train_data), desc="TrainingSequences") as pbar:
    knn_classifier_forward.fit(np.array(X_train), y_train)
    pbar.update(len(X_train_data))

joblib.dump(knn_classifier_forward, 'knn_model.pkl')
# Predict on training set for forward sequences
y_pred_train_forward = knn_classifier_forward.predict(np.array(X_train))
accuracy_train_forward = accuracy_score(y_train, y_pred_train_forward)

y_pred_val_forward = knn_classifier_forward.predict(np.array(X_val))
accuracy_val_forward = accuracy_score(y_val, y_pred_val_forward)



print("Training Accuracy :", accuracy_train_forward)
print("Val Accuracy :", accuracy_val_forward)

new_seq = []
for count,  i in enumerate(range(len(de_novo_sequence) - 11 + 1)):
    kmer = de_novo_sequence[i:i + 11]
    new_seq = new_seq + [kmer]

while "-" in de_novo_sequence:
    keys_with_one_dash = [key for key in new_seq if key.count('-') == 1]
    for k in keys_with_one_dash:
            if k in de_novo_sequence:
                # Convert de novo sequence and its reverse to numeric representation
                de_novo_sequence_numeric = sequence_to_numeric(k)

                X_de_novo = np.array([de_novo_sequence_numeric])
                # Make predictions for the de novo sequence reverse
                y_pred_de_novo = knn_classifier_forward.predict(X_de_novo)
                # Convert the predicted labels back to sequences for verification
                predicted_sequence = pred_dict[y_pred_de_novo[0]]
                index = de_novo_sequence.index(k)
                index1 = k.index("-")
                if 0 <= (index + index1) < len(de_novo_sequence):
                    de_novo_sequence = de_novo_sequence[:(index + index1)] + predicted_sequence[
                        index1] + de_novo_sequence[(index + index1) + 1:]

            # Update new_seq after filling a gap
            new_seq.clear()
            for count, i in enumerate(range(len(de_novo_sequence) - 11 + 1)):
                kmer = de_novo_sequence[i:i + 11]
                new_seq = new_seq + [kmer]
            keys_with_one_dash = [key for key in new_seq if key.count('-') == 1]

# Print the predicted sequence for the de novo sequence
print("Predicted Sequence for De Novo (forward):", de_novo_sequence)