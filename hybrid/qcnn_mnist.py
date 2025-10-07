import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pennylane as qml
from pennylane import numpy as np

import matplotlib.pyplot as plt

# Set a seed for reproducibility
torch.manual_seed(42)

# --- Quantum Circuit Definitions ---

# Define the number of qubits and the quantum device for simulation
N_QUBITS = 16  # We'll use a 4x4 image, so 16 pixels -> 16 qubits
DEV = qml.device("default.qubit", wires=N_QUBITS)

def get_conv_and_pool_wires():
    """Generates the wire mappings for the convolutional and pooling layers."""
    # This defines the 2x2 patches for the first convolution
    conv1_wires = [
        [0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]
    ]
    # We pool by measuring the last 3 qubits in each patch and keeping the first one
    pool1_wires_to_measure = [1, 4, 5, 2, 6, 7, 9, 12, 13, 11, 14, 15]
    pool1_wires_to_keep = [0, 3, 8, 10]
    # The second convolution acts on the qubits that were kept
    conv2_wires = [[0, 3, 8, 10]]
    
    return conv1_wires, pool1_wires_to_measure, pool1_wires_to_keep, conv2_wires

CONV1_WIRES, POOL1_MEASURE, POOL1_KEEP, CONV2_WIRES = get_conv_and_pool_wires()

def quantum_conv_filter(weights, wires):
    """A parameterized quantum circuit representing the convolutional filter."""
    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])
    qml.RY(weights[2], wires=wires[2])
    qml.RY(weights[3], wires=wires[3])
    
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.CNOT(wires=[wires[3], wires[0]])

def quantum_pool_filter(wires_to_measure, wires_to_keep):
    """
    The pooling layer. We simply measure qubits to reduce dimensionality.
    In a more advanced circuit, measurement outcomes could control gates on `wires_to_keep`.
    """
    for wire in wires_to_measure:
        qml.measure(wire)

@qml.qnode(DEV, interface="torch", diff_method="parameter-shift")
def qcnn_circuit(inputs, conv1_weights, conv2_weights):
    """The main QCNN circuit."""
    # 1. Data Encoding: Encode the 4x4 image pixels as rotation angles
    for i in range(N_QUBITS):
        qml.RY(np.pi * inputs[i], wires=i)

    # 2. First Convolutional Layer
    for wires in CONV1_WIRES:
        quantum_conv_filter(conv1_weights, wires)
        
    # 3. First Pooling Layer
    # Here, we implicitly "pool" by simply not using the measured qubits anymore.
    # The `qml.measure` call in a simulator effectively removes them from subsequent calculations.
    # quantum_pool_filter(POOL1_MEASURE, POOL1_KEEP)

    # 4. Second Convolutional Layer
    for wires in CONV2_WIRES:
        quantum_conv_filter(conv2_weights, wires)
        
    # 5. Measurement: Return the expectation value of the remaining qubits
    return [qml.expval(qml.PauliZ(w)) for w in POOL1_KEEP]

class HybridQCNN(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network for image classification.
    """
    def __init__(self, num_classes=10):
        super(HybridQCNN, self).__init__()
        
        # Define the quantum layers and their weights
        # Each conv filter has 4 parameters
        conv1_weight_shapes = {"conv1_weights": 4}
        conv2_weight_shapes = {"conv2_weights": 4}
        
        # Create the quantum part as a TorchLayer
        self.q_layer = qml.qnn.TorchLayer(qcnn_circuit, {
            "conv1_weights": conv1_weight_shapes["conv1_weights"],
            "conv2_weights": conv2_weight_shapes["conv2_weights"]
        })
        
        # Create the classical part
        self.c_layer = nn.Linear(len(POOL1_KEEP), num_classes)

    def forward(self, x):
        # The input x is a batch of images
        # We process one image at a time
        activations = []
        for image in x:
            # Flatten the image to a 1D vector for the quantum circuit
            flat_image = torch.flatten(image)
            q_out = self.q_layer(flat_image)
            activations.append(q_out)
        
        # Stack the results for the batch
        q_out_batch = torch.stack(activations)
        
        # Pass through the classical layer
        c_out = self.c_layer(q_out_batch)
        return c_out

# --- Training and Evaluation ---

def main():
    """Main function to train and test the QCNN model."""
    # Hyperparameters
    batch_size = 16
    epochs = 5
    learning_rate = 0.01

    # Data loading and pre-processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((4, 4))  # Downscale images to 4x4
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model, loss function, and optimizer
    model = HybridQCNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss and backpropagate
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        print(f"Epoch {epoch+1} average loss: {running_loss / len(train_loader):.4f}")

    print("Training finished.")

    # --- Evaluation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f} %')
    
    # --- Visualize a prediction ---
    images, labels = next(iter(test_loader))
    output = model(images[0].unsqueeze(0))
    _, prediction = torch.max(output.data, 1)

    plt.imshow(images[0].squeeze(), cmap="gray")
    plt.title(f"Prediction: {prediction.item()} - Actual: {labels[0].item()}")
    plt.show()


if __name__ == '__main__':
    # Note: Training will be slow due to quantum circuit simulation.
    # For a real application, you'd limit the number of training samples
    # or use a much more efficient simulator/real hardware.
    main()