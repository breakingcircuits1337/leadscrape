import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

# -----------------------------------------------------------
# 1. Define the MODIFIED Quantum Recurrent Cell
# -----------------------------------------------------------

class QRNNCell(nn.Module):
    """
    Quantum Recurrent Neural Network Cell with a sophisticated,
    strongly entangling circuit architecture.
    """
    def __init__(self, input_size, hidden_size, n_qubits=4, n_layers=2):
        """
        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
                              Must be equal to n_qubits.
            n_qubits (int): The number of qubits in the quantum circuit.
            n_layers (int): The number of layers in the strongly entangling circuit.
        """
        super(QRNNCell, self).__init__()
        
        if hidden_size != n_qubits:
            raise ValueError(f"hidden_size ({hidden_size}) must be equal to "
                             f"n_qubits ({n_qubits}).")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        # Define the quantum device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # ### SOPHISTICATED CIRCUIT CHANGE ###
        # Automatically determine the shape of the weights tensor required
        # by the StronglyEntanglingLayers template.
        self.weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, 
                                                                n_wires=self.n_qubits)
        # Initialize the weights as a learnable PyTorch parameter
        self.weights = nn.Parameter(torch.randn(self.weights_shape) * 0.01)

        # Define the quantum node
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def quantum_circuit(inputs, weights):
            """The PQC with a strongly entangling architecture."""
            # Encode classical input data into quantum state
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))

            # ### SOPHISTICATED CIRCUIT CHANGE ###
            # Apply the sophisticated, trainable circuit.
            # This single line replaces the previous manual for-loops.
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # Wrap the QNode in a TorchLayer
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, 
                                                weight_shapes={"weights": self.weights_shape})

    def forward(self, x, hidden_state):
        """
        Forward pass of the QRNN cell.
        """
        if x.shape[1] != self.n_qubits:
             raise ValueError(f"Input feature size ({x.shape[1]}) does not match "
                              f"n_qubits ({self.n_qubits}).")

        new_hidden = self.quantum_layer(x)
        return new_hidden

# -----------------------------------------------------------
# 2. The Full QRNN Model (No changes needed here)
# -----------------------------------------------------------

class QRNN(nn.Module):
    """
    A full QRNN model for sequence processing. This class does not need
    to be changed, demonstrating the modularity of the cell.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, n_qubits, n_q_layers):
        super(QRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.project_in = nn.Linear(embedding_dim, hidden_size)
        
        # Instantiate the new, more powerful cell
        self.qrnn_cell = QRNNCell(input_size=hidden_size, hidden_size=hidden_size, 
                                  n_qubits=n_qubits, n_layers=n_q_layers)
        
        self.project_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        batch_size = input_sequence.shape[0]
        seq_length = input_sequence.shape[1]
        
        hidden = torch.zeros(batch_size, self.hidden_size)
        embedded = self.embedding(input_sequence)
        
        for t in range(seq_length):
            input_t = embedded[:, t, :]
            projected_t = torch.tanh(self.project_in(input_t))
            hidden = self.qrnn_cell(projected_t, hidden)

        output = self.project_out(hidden)
        return output

# -----------------------------------------------------------
# 3. Small Character-Level Dataset Example
# -----------------------------------------------------------

class CharSequenceDataset(torch.utils.data.Dataset):
    """
    Synthetic character-level sequence classification dataset.

    - Vocabulary: lowercase letters (26)
    - Label: (number of vowels in sequence) mod NUM_CLASSES
    """
    def __init__(self, num_samples=256, seq_len=8, num_classes=4, seed=42):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.seq_len = seq_len
        self.num_classes = num_classes

        # Build vocab and mappings
        self.vocab = [chr(ord('a') + i) for i in range(26)]
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.vowels = set(list("aeiou"))

        # Generate samples
        X = []
        y = []
        for _ in range(num_samples):
            chars = torch.randint(0, len(self.vocab), (seq_len,), generator=rng)
            # Count vowels in the sequence
            vowel_count = sum(1 for idx in chars.tolist() if self.vocab[idx] in self.vowels)
            label = vowel_count % num_classes
            X.append(chars)
            y.append(label)
        self.X = torch.stack(X)  # [N, T]
        self.y = torch.tensor(y, dtype=torch.long)  # [N]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


if __name__ == '__main__':
    # --- Model Hyperparameters ---
    VOCAB_SIZE = 26
    EMBEDDING_DIM = 8
    N_QUBITS = 4
    HIDDEN_SIZE = N_QUBITS
    QUANTUM_LAYERS = 2
    NUM_CLASSES = 4

    # --- Data Hyperparameters ---
    TRAIN_SAMPLES = 256
    TEST_SAMPLES = 64
    SEQ_LENGTH = 8
    BATCH_SIZE = 8
    EPOCHS = 5
    LR = 0.01

    # Build datasets and loaders
    train_ds = CharSequenceDataset(num_samples=TRAIN_SAMPLES, seq_len=SEQ_LENGTH, num_classes=NUM_CLASSES, seed=123)
    test_ds = CharSequenceDataset(num_samples=TEST_SAMPLES, seq_len=SEQ_LENGTH, num_classes=NUM_CLASSES, seed=456)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing the QRNN with a Strongly Entangling circuit...")
    model = QRNN(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        output_size=NUM_CLASSES,
        n_qubits=N_QUBITS,
        n_q_layers=QUANTUM_LAYERS
    )
    print(model)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\n--- Training on synthetic character-level dataset ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_ds)
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch:02d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.3f}")

    print("\n--- Demonstration Complete ---")