"""
Runner script for the Hybrid Quantum-Temporal Network on IBM Quantum (or locally via Aer).

Usage:
    python -m hybrid.run_ibm

Notes:
- Ensure you have IBM Quantum credentials configured if you plan to use hardware/runtime.
- Defaults to local Aer simulation if IBM runtime is not available.
"""

import torch
from hybrid_quantum_tpm import HybridQuantumTemporalNetwork, HAS_IBM_RUNTIME


def main():
    # Example synthetic input: [batch, time, features]
    batch = 2
    time = 50
    features = 10
    X = torch.randn(batch, time, features)

    # Initialize hybrid network
    hybrid = HybridQuantumTemporalNetwork(
        total_qubits=133,
        input_dim=features,
        manifold_dim=64,
        num_frequencies=8,
        use_time_warping=True,
        task_type='entropy',
    )

    if HAS_IBM_RUNTIME:
        print("Running on IBM Quantum Runtime...")
        best_params, history = hybrid.run_on_ibm(
            X,
            backend_name='ibm_brisbane',
            num_iterations=5,
            population_size=4,
            shots=512
        )
        print("Training complete.")
        print(f"Best fitness history: {history}")
    else:
        print("IBM Runtime not available. Running local Aer quick test...")
        counts, fitness = hybrid.quick_test(X)
        print(f"Local fitness: {fitness:.4f}")
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for state, count in top:
            print(f"{state[:16]}...{state[-16:]}: {count}")


if __name__ == "__main__":
    main()