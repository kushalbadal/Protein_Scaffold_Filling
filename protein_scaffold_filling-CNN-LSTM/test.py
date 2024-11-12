import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Function to convert sequences to numeric representation
def sequence_to_numeric(sequence):
    return [CHAR_TO_INT[char] for char in sequence]

# Function to convert numeric representation back to sequence
def numeric_to_sequence(numeric_seq):
    return "".join([INT_TO_CHAR[num] for num in numeric_seq])

# Function to extract sequences from a FASTA file
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

# Load training sequences
sequences_to_train_on = 100
training_sequences = get_sequences("data/training_sequences.txt")
training_sequences = training_sequences[:sequences_to_train_on]
training_sequences_reversed = training_sequences[::-1]
print(training_sequences_reversed)

#Load Target sequences
target_sequence = get_sequences("data/target_sequence.txt")[0]

# Load de novo sequence and its reverse
de_novo_sequence = get_sequences("data/de_novo_sequence.txt")[0]
de_novo_sequence_reversed = de_novo_sequence[::-1]
print(de_novo_sequence_reversed)

# Create CHAR_TO_INT and INT_TO_CHAR dictionaries
all_chars = set("".join(training_sequences) + de_novo_sequence + target_sequence)
NUM_CLASSES = len(all_chars)
CHAR_TO_INT = {c: i for i, c in enumerate(all_chars, start=1)}
INT_TO_CHAR = {v: k for k, v in CHAR_TO_INT.items()}

# Convert training sequences to numeric representation
training_sequences_numeric = [sequence_to_numeric(seq) for seq in training_sequences]
training_sequences_reversed_numeric = [sequence_to_numeric(seq) for seq in training_sequences_reversed]

# Convert de novo sequence and its reverse to numeric representation
de_novo_sequence_numeric = sequence_to_numeric(de_novo_sequence)
de_novo_sequence_reversed_numeric = sequence_to_numeric(de_novo_sequence_reversed)

# Find the maximum length of sequences
max_seq_length = max(len(seq) for seq in training_sequences_numeric)

# Pad sequences to the same length
padded_sequences_numeric = [seq + [0] * (max_seq_length - len(seq)) for seq in training_sequences_numeric]
padded_sequences_rev_numeric = [seq + [0] * (max_seq_length - len(seq)) for seq in training_sequences_reversed_numeric]

# Convert padded sequences to a NumPy array
X_train = np.array(padded_sequences_numeric)
y_train = np.arange(sequences_to_train_on)  # Assuming each sequence has a label


# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    np.array(padded_sequences_numeric),
    np.arange(sequences_to_train_on),
    test_size=0.1,  # 10% of the data for validation
    random_state=42
)
# Train the KNN classifier for forward sequences
k = 3
knn_classifier_forward = KNeighborsClassifier(n_neighbors=k )

# Use tqdm to visualize training progress
with tqdm(total=X_train.shape[0], desc="Training Forward Sequences") as pbar:
    knn_classifier_forward.fit(X_train, y_train)
    pbar.update(X_train.shape[0])

# Predict on validation set
y_pred_val_forward = knn_classifier_forward.predict(X_val)
accuracy_val_forward = accuracy_score(y_val, y_pred_val_forward)
y_pred_train_forward = knn_classifier_forward.predict(X_train)
accuracy_train_forward = accuracy_score(y_train, y_pred_train_forward)

print("Validation Accuracy for Forward Sequences:", accuracy_val_forward)
print("Training Accuracy for Forward Sequences:", accuracy_train_forward)

# Train the KNN classifier for reverse sequences
knn_classifier_reverse = KNeighborsClassifier(n_neighbors=k)

# Use tqdm to visualize training progress
with tqdm(total=X_train.shape[0], desc="Training Reverse Sequences") as pbar:
    knn_classifier_reverse.fit(X_train, y_train)  # Use X_train and y_train from reverse sequences
    pbar.update(X_train.shape[0])

# Predict on validation set
y_pred_val_reverse = knn_classifier_reverse.predict(X_val)
accuracy_val_reverse = accuracy_score(y_val, y_pred_val_reverse)
y_pred_train_reverse = knn_classifier_reverse.predict(X_train)
accuracy_train_reverse = accuracy_score(y_train, y_pred_train_reverse)

print("Validation Accuracy for Reverse Sequences:", accuracy_val_reverse)
print("Training Accuracy for Reverse Sequences:", accuracy_train_reverse)

padded_de_novo_numeric = de_novo_sequence_numeric + [0] * (max_seq_length - len(de_novo_sequence_numeric))
X_de_novo = np.array([padded_de_novo_numeric])

# Make predictions for the de novo sequence reverse
y_pred_de_novo_forward= knn_classifier_forward.predict(X_de_novo)

padded_de_novo_rev_numeric = de_novo_sequence_reversed_numeric + [0] * (max_seq_length - len(de_novo_sequence_reversed_numeric))
X_de_novo = np.array([padded_de_novo_rev_numeric])

# Make predictions for the de novo sequence reverse
y_pred_de_novo_rev = knn_classifier_reverse.predict(X_de_novo)



# Convert the predicted labels back to sequences for verification
predicted_sequence_forward = numeric_to_sequence(training_sequences_numeric[y_pred_de_novo_forward[0]])
predicted_sequence_reverse = numeric_to_sequence(training_sequences_reversed_numeric[y_pred_de_novo_rev[0]])

# Print the predicted sequences for the de novo sequence and its reverse
print("Predicted Sequence for De Novo (forward):", predicted_sequence_forward)
print("Predicted Sequence for De Novo (reverse):", predicted_sequence_reverse)



