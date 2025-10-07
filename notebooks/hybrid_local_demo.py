# %% [markdown]
# # Hybrid Quantum-Temporal Local Demo (Jupytext)
#
# Local-only quick tests (no IBM Runtime), demonstrating:
# - Cognitive circuit (region-aware parameter mapping)
# - Neuromodulation with affective-cognitive fusion
# - Fast time-manifold circuit (TPM-driven theta/tau)
#
# Requirements:
# ```
# pip install torch torchvision matplotlib networkx pennylane qiskit qiskit-aer jupytext
# ```
#
# Convert to ipynb:
# ```
# jupytext --to notebook notebooks/hybrid_local_demo.py
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

import matplotlib.pyplot as plt

# %matplotlib inline

# %% [markdown]
# ## 1. Cognitive Circuit (Region-Aware Parameters)
# TPM features mapped to region parameters, evaluated locally via Aer.

# %%
# Random input batch [B, T, F]
B, T, F = 2, 50, 10
X = torch.randn(B, T, F)

hybrid = HybridQuantumTemporalNetwork(
    total_qubits=133,
    input_dim=F,
    manifold_dim=64,
    task_type='decision',
    use_neuromodulation=False,
    use_fast_time_manifold=False,
)

counts, fitness = hybrid.quick_test(X, num_choices=4)
print("Local decision task - fitness:", fitness)

# Optional: visualize top outcomes
top_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
print("Top states:")
for s, c in top_items:
    print(s[:15], "...", s[-15:], c)

# %% [markdown]
# ## 2. Neuromodulation with Affective-Cognitive Fusion
# Blend TPM-driven parameters with affect-defined modulation strengths.

# %%
hybrid_nm = HybridQuantumTemporalNetwork(
    total_qubits=133,
    input_dim=F,
    manifold_dim=64,
    task_type='inhibition',
    use_neuromodulation=True
)

emotion = {'fear': 0.7, 'executive_control': 0.4, 'interoception': 0.3}
drives  = {'threat_vigilance': 0.6, 'reward_seeking': 0.2, 'hunger': 0.5}
hybrid_nm.set_affect_state(emotion, drives, alpha=0.5)

counts_nm, fitness_nm = hybrid_nm.quick_test(
    X,
    target_pattern='1010',
    distractor_patterns=['0101','1111']
)
print("Local neuromodulation (inhibition) - fitness:", fitness_nm)

# %% [markdown]
# ## 3. Fast Time-Manifold Circuit (TPM-driven Theta/Tau)
# The fast circuit uses TPM to produce both theta angles and tau time-scaling.

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
# ## 4. Quick Plot Utilities (Optional)

# %%
def plot_counts_distribution(counts, title="Counts Distribution", max_states=20):
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:max_states]
    labels = [k[-16:] for k, _ in items]
    values = [v for _, v in items]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_counts_distribution(counts, title="Cognitive circuit (decision)")
plot_counts_distribution(counts_nm, title="Neuromodulation (inhibition)")
plot_counts_distribution(counts_tm, title="Time-manifold (predictive)")

# %%
print("Local hybrid demo complete.")