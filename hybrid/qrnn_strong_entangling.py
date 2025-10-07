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
# 3. Example Usage
# -----------------------------------------------------------
if __name__ == '__main__':
    # --- Model Hyperparameters ---
    VOCAB_SIZE = 1000
    EMBEDDING_DIM = 12
    N_QUBITS = 6
    HIDDEN_SIZE = N_QUBITS
    QUANTUM_LAYERS = 3  # Number of layers in the sophisticated circuit
    OUTPUT_SIZE = 10
    
    # --- Data Hyperparameters ---
    BATCH_SIZE = 5
    SEQ_LENGTH = 10

    print("Initializing the QRNN with a Strongly Entangling circuit...")
    model = QRNN(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        n_qubits=N_QUBITS,
        n_q_layers=QUANTUM_LAYERS
    )
    print(model)
    
    # --- Create Dummy Data ---
    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
    dummy_labels = torch.randint(0, OUTPUT_SIZE, (BATCH_SIZE,))
    
    # --- Training Loop Example ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("\n--- Starting a simple training demonstration ---")
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_labels)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1:02d}, Loss: {loss.item():.4f}")

    print("\n--- Demonstration Complete ---")