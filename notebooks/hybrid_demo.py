# %% [markdown]
# # Hybrid Quantum-Temporal Demo (Jupytext)
#
# This notebook (in percent-script format) demonstrates:
# - Running the hybrid Temporal Phase Manifold (TPM) + Quantum circuits locally
# - Neuromodulation with affective-cognitive fusion
# - Fast time-manifold circuit driven by TPM (theta/tau)
# - Optional multi-objective optimization on IBM Runtime (with ES and CMA-ES)
#
# Requirements (install in a cell or your environment):
# ```
# pip install torch torchvision matplotlib networkx pennylane qiskit qiskit-aer qiskit-ibm-runtime jupytext
# ```
#
# To convert to ipynb:
# ```
# jupytext --to notebook notebooks/hybrid_demo.py
# ```

# %% [markdown]
# ## 0. Environment and Imports

# %%
import os, sys
sys.path.append(os.path.abspath("."))

import numpy as np
import torch

# Hybrid pipeline and modules
from hybrid.hybrid_quantum_tpm import HybridQuantumTemporalNetwork
from hybrid.quantum_neuromodulation import StateDependentQuantumNeuromodulation, create_neuromodulated_circuit
from hybrid.quantum_brain_cognitive import FastQuantumBrain
from hybrid.affective_cognitive import emotion_drive_to_neuromod_angles

# Visualization (optional)
import matplotlib.pyplot as plt

# Enable inline plots if running as notebook
# %matplotlib inline

# %% [markdown]
# ## 1. Cognitive Circuit (Region-Aware Parameters)
# Run the hybrid pipeline in standard cognitive mode: TPM features map to each brain region's parameters and we evaluate a cognitive task locally.

# %%
# Random input batch [B, T, F]
B, T, F = 2, 50, 10
X = torch.randn(B, T, F)

hybrid = HybridQuantumTemporalNetwork(
    total_qubits=133,
    input_dim=F,
    manifold_dim=64,
    task_type='decision',     # try: 'entropy', 'decision', 'spatial', ...
    use_neuromodulation=False,
    use_fast_time_manifold=False,
)

counts, fitness = hybrid.quick_test(X, num_choices=4)
print("Local decision task - fitness:", fitness)

# %% [markdown]
# ## 2. Neuromodulation Mode with Affective-Cognitive Fusion
# Blend TPM-driven neuromodulation with explicit affective state (emotions + drives).  
# alpha in [0..1] controls the blend: 1.0 = TPM only, 0.0 = affect only.

# %%
hybrid_nm = HybridQuantumTemporalNetwork(
    total_qubits=133,
    input_dim=F,
    manifold_dim=64,
    task_type='inhibition',
    use_neuromodulation=True
)

# Affective state
emotion = {'fear': 0.7, 'executive_control': 0.4, 'interoception': 0.3}
drives  = {'threat_vigilance': 0.6, 'reward_seeking': 0.2, 'hunger': 0.5}
hybrid_nm.set_affect_state(emotion, drives, alpha=0.5)  # 50% TPM + 50% affect

counts_nm, fitness_nm = hybrid_nm.quick_test(
    X,
    target_pattern='1010',
    distractor_patterns=['0101','1111']
)
print("Local neuromodulation (inhibition) - fitness:", fitness_nm)

# %% [markdown]
# ## 3. Fast Time-Manifold Circuit (Theta/Tau from TPM)
# Use the trainable time-manifold cognitive circuit. TPM features generate both theta and tau (time scaling) parameters.

# %%
hybrid_tm = HybridQuantumTemporalNetwork(
    total_qubits=133,
    input_dim=F,
    manifold_dim=64,
    task_type='predictive',
    use_fast_time_manifold=True,
    fast_depth=2
)

counts_tm, fitness_tm = hybrid_tm.quick_test(
    X,
    expected_pattern='10110110',
    prediction_error_threshold=0.3
)
print("Local time-manifold (predictive) - fitness:", fitness_tm)

# %% [markdown]
# ## 4. Multi-Objective Optimization on IBM Runtime (Optional)
# Requires IBM Quantum credentials. This example uses CMA-ES and prints step-size and covariance eigenvalue summary each iteration.

# %%
USE_IBM = False  # set True if your IBM Quantum account is configured
if USE_IBM:
    objectives = {'inhibition': 0.4, 'flexibility': 0.3, 'attention': 0.3}
    best_params, history = hybrid_nm.run_on_ibm(
        X,
        backend_name='ibm_brisbane',       # choose an available backend
        num_iterations=6,
        population_size=8,
        shots=512,
        resilience_level=3,
        use_dynamic_decoupling=True,
        objectives=objectives,
        use_cma_es=True,                   # enable CMA-ES
        cma_popmult=1.5,                   # scale population size
        target_pattern='1010',
        distractor_patterns=['0101','1111'],
        num_contexts=4,
        focus_region_size=8
    )
    print("IBM run - final fitness:", history[-1])

# %% [markdown]
# ## 5. QCNN (MNIST) and QRNN (Character-Level) Demos (Optional)
# You can run the standalone demos from their modules or copy code into cells for fine-grained control.
# - QCNN: hybrid/qcnn_mnist.py (PennyLane + PyTorch)
# - QRNN: hybrid/qrnn_strong_entangling.py (PennyLane + TorchLayer)
#
# Running directly as scripts:
# ```
# %run -m hybrid.qcnn_mnist
# %run -m hybrid.qrnn_strong_entangling
# ```

# %%
print("Hybrid demo complete. You can tweak tasks, alpha, and objectives above.")