import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm

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
# Function to extract sequences
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
    new_seq = []
    for amino in sequences:
        for i in range(len(amino) - 7 + 1):
            kmer = amino[i:i + 7]
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
    return dict, dict_for_prediction

# Load training sequences
training_sequences, dict_for_prediction = get_sequences("data/training_sequences.txt")
sequences_to_train_on = len(training_sequences)

de_novo_sequence = "MTQSPSSLSASVGDRVTITCK---"
all_chars = set("".join(training_sequences) + de_novo_sequence)
NUM_CLASSES = len(all_chars)
CHAR_TO_INT = {c: i for i, c in enumerate(all_chars, start=1)}
INT_TO_CHAR = {v: k for k, v in CHAR_TO_INT.items()}
# Convert training sequences to numeric representation for KNN
X_train_data = []
y_train_data = []
for keys, values in training_sequences.items():
    X_train_data = X_train_data + [sequence_to_numeric(keys)]
    y_train_data = y_train_data + [values]


max_seq_length = max(len(seq) for seq in X_train_data)
# Standardize feature values
scaler = StandardScaler()
X_train_data_standardized = scaler.fit_transform(X_train_data)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_data_standardized, y_train_data, test_size=0.1, random_state=42)

# Train the KNN classifier for forward sequences
k = 3  # Experiment with different values of k
knn_classifier_forward = KNeighborsClassifier(n_neighbors=k)

# Use tqdm to visualize training progress
with tqdm(total=len(X_train_data_standardized), desc="Training Forward Sequences") as pbar:
    knn_classifier_forward.fit(np.array(X_train), np.array(y_train))
    pbar.update(len(X_train_data_standardized))

# Predict on validation set for forward sequences
y_pred_val_forward = knn_classifier_forward.predict(np.array(X_val))
accuracy_val_forward = accuracy_score(y_val, y_pred_val_forward)

print("Validation Accuracy for Forward Sequences:", accuracy_val_forward)

new_seq = []
for count,  i in enumerate(range(len(de_novo_sequence) - 7 + 1)):
    kmer = de_novo_sequence[i:i + 7]
    new_seq = new_seq + [kmer]

while "-" in de_novo_sequence:
    keys_with_one_dash = [key for key in new_seq if key.count('-') == 1]
    if len(keys_with_one_dash) == 0:
        keys_with_one_dash = [key for key in new_seq if key.count('-') == 2]
    for k in keys_with_one_dash:
        if k in de_novo_sequence:
            # Convert de novo sequence and its reverse to numeric representation
            de_novo_sequence_numeric = sequence_to_numeric(k)
            padded_de_novo_numeric = de_novo_sequence_numeric + [0] * (max_seq_length - len(de_novo_sequence_numeric))

            X_de_novo = np.array([padded_de_novo_numeric])
            # Make predictions for the de novo sequence reverse
            y_pred_de_novo_forward = knn_classifier_forward.predict(X_de_novo)
            # Convert the predicted labels back to sequences for verification
            predicted_sequence_forward = numeric_to_sequence(X_train_data[y_pred_de_novo_forward[0]])
            predicted_word = dict_for_prediction[training_sequences[predicted_sequence_forward]]
            print(predicted_word)
            index = de_novo_sequence.index(k)
            index1 = k.index("-")
            if 0 <= (index + index1) < len(de_novo_sequence):
                # de_novo_sequence = de_novo_sequence[:(index + index1)] + predicted_word[index1] + de_novo_sequence[(index + index1) + 1:]
                de_novo_sequence = de_novo_sequence.replace(k, predicted_word)
            print(de_novo_sequence)
        # Update new_seq after filling a gap
        new_seq.clear()
        for count, i in enumerate(range(len(de_novo_sequence) - 7 + 1)):
            kmer = de_novo_sequence[i:i + 7]
            new_seq = new_seq + [kmer]
        keys_with_one_dash = [key for key in new_seq if key.count('-') == 1]

# Print the predicted sequence for the de novo sequence
print("Predicted Sequence for De Novo (forward):", de_novo_sequence)
