"""
Hybrid Quantum-Temporal Network for IBM Quantum

This module integrates:
- TemporalPhaseManifold (TPM): a PyTorch model that produces phase-based temporal features
- FastQuantumBrain (FQB): a structured Qiskit circuit representing brain regions

The hybrid network uses TPM outputs to modulate parameters of the quantum circuit,
creating a single end-to-end system suitable for execution on IBM Quantum backends.

Requirements:
- torch
- qiskit
- qiskit-aer (for local simulation)
- qiskit-ibm-runtime (for IBM hardware/runtime)

Usage:
    from hybrid_quantum_tpm import HybridQuantumTemporalNetwork

    hybrid = HybridQuantumTemporalNetwork(
        total_qubits=133,
        input_dim=10,
        manifold_dim=64,
        num_frequencies=8,
        use_time_warping=True,
        task_type='entropy'  # or 'decision', 'pattern', 'memory', 'classification', 'spatial'
    )

    # Example data [batch, time, features]
    import torch
    X = torch.randn(2, 50, 10)

    # Run locally (Aer) to validate
    counts, fitness = hybrid.quick_test(X)

    # Or run on IBM Quantum
    best_params, history = hybrid.run_on_ibm(
        X,
        backend_name='ibm_brisbane',
        num_iterations=5,
        population_size=4,
        shots=512
    )
"""

import math
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes, EfficientSU2, TwoLocal
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimator

try:
    # Optional IBM runtime imports (only needed for hardware runs)
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Options
    HAS_IBM_RUNTIME = True
except Exception:
    HAS_IBM_RUNTIME = False


# -------------------------------
# Temporal Phase Manifold (TPM)
# -------------------------------

class TemporalPhaseEmbedding(nn.Module):
    def __init__(self, input_dim: int, manifold_dim: int = 64, num_frequencies: int = 8, learnable_phase: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.manifold_dim = manifold_dim
        self.num_frequencies = num_frequencies

        self.linear_projection = nn.Linear(input_dim, manifold_dim)

        if learnable_phase:
            self.frequencies = nn.Parameter(torch.logspace(-1, 2, num_frequencies))
        else:
            self.register_buffer('frequencies', torch.logspace(-1, 2, num_frequencies))

        self.amplitude_network = nn.Sequential(
            nn.Linear(manifold_dim, num_frequencies),
            nn.Softmax(dim=-1)
        )

        self.phase_offset = nn.Parameter(torch.zeros(num_frequencies))

    def forward(self, x: torch.Tensor, time_coords: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        manifold_repr = self.linear_projection(x)  # [B, T, D]

        if time_coords is None:
            time_coords = torch.arange(seq_len, dtype=x.dtype, device=x.device)
            time_coords = time_coords.unsqueeze(0).expand(batch_size, -1)

        time_coords = time_coords / seq_len * 2 * math.pi  # normalize to [0, 2π]
        amplitudes = self.amplitude_network(manifold_repr)  # [B, T, F]

        phase_coords = (
            time_coords.unsqueeze(-1) * self.frequencies.unsqueeze(0).unsqueeze(0) +
            self.phase_offset.unsqueeze(0).unsqueeze(0)
        )  # [B, T, F]

        real_part = amplitudes * torch.cos(phase_coords)
        imag_part = amplitudes * torch.sin(phase_coords)
        phase_embedding = manifold_repr.unsqueeze(-1) * torch.complex(real_part, imag_part)  # [B, T, D, F]

        return phase_embedding, phase_coords


class ManifoldAttention(nn.Module):
    def __init__(self, manifold_dim: int, num_heads: int = 8, num_frequencies: int = 8):
        super().__init__()
        self.manifold_dim = manifold_dim
        self.num_heads = num_heads
        self.num_frequencies = num_frequencies
        self.head_dim = manifold_dim // num_heads
        assert manifold_dim % num_heads == 0

        self.q_proj = nn.Linear(manifold_dim, manifold_dim)
        self.k_proj = nn.Linear(manifold_dim, manifold_dim)
        self.v_proj = nn.Linear(manifold_dim, manifold_dim)
        self.out_proj = nn.Linear(manifold_dim, manifold_dim)

        self.phase_bias = nn.Parameter(torch.zeros(num_heads, num_frequencies))

    def forward(self, x: torch.Tensor, phase_coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        phase_diff = phase_coords.unsqueeze(2) - phase_coords.unsqueeze(1)  # [B, T, T, F]
        phase_coherence = torch.cos(phase_diff)
        weighted_coherence = (
            phase_coherence.unsqueeze(1) * self.phase_bias.view(1, self.num_heads, 1, 1, self.num_frequencies)
        ).sum(dim=-1)  # [B, H, T, T]

        attn_scores = attn_scores + weighted_coherence

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.manifold_dim)
        return self.out_proj(attended)


class TemporalWarpingLayer(nn.Module):
    def __init__(self, manifold_dim: int):
        super().__init__()
        self.warp_predictor = nn.Sequential(
            nn.Linear(manifold_dim, manifold_dim // 2),
            nn.ReLU(),
            nn.Linear(manifold_dim // 2, 3)  # [stretch, shift, reverse_prob]
        )

    def forward(self, x: torch.Tensor, time_coords: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        warp_params = self.warp_predictor(x)
        stretch = torch.sigmoid(warp_params[..., 0]) * 2  # approx [0, 2]
        shift = torch.tanh(warp_params[..., 1]) * seq_len * 0.1
        reverse_prob = torch.sigmoid(warp_params[..., 2])

        warped_times = time_coords * stretch + shift
        if self.training:
            forward_times = warped_times
            reverse_times = seq_len - warped_times
            warped_times = reverse_prob.unsqueeze(-1) * reverse_times + (1 - reverse_prob.unsqueeze(-1)) * forward_times

        warped_times = torch.cummax(warped_times, dim=1)[0]  # enforce monotonicity
        return warped_times


class TemporalPhaseManifold(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        manifold_dim: int = 64,
        num_frequencies: int = 8,
        num_conv_layers: int = 2,
        num_attention_layers: int = 2,
        num_heads: int = 8,
        use_time_warping: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.manifold_dim = manifold_dim
        self.use_time_warping = use_time_warping

        self.phase_embedding = TemporalPhaseEmbedding(input_dim, manifold_dim, num_frequencies)
        self.attentions = nn.ModuleList([ManifoldAttention(manifold_dim, num_heads, num_frequencies) for _ in range(num_attention_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(manifold_dim) for _ in range(num_attention_layers)])

        if use_time_warping:
            self.time_warping = TemporalWarpingLayer(manifold_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(manifold_dim, manifold_dim // 2),
            nn.ReLU(),
            nn.Linear(manifold_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor, return_phase_info: bool = False):
        batch_size, seq_len, _ = x.shape
        phase_embedding, phase_coords = self.phase_embedding(x)
        phase_features = phase_embedding.abs().mean(dim=-1)  # [B, T, D]

        for idx, attention in enumerate(self.attentions):
            residual = phase_features
            phase_features = attention(phase_features, phase_coords)
            phase_features = self.layer_norms[idx](phase_features + residual)

        if self.use_time_warping:
            time_coords = torch.arange(seq_len, dtype=x.dtype, device=x.device).unsqueeze(0).expand(batch_size, -1)
            warped_times = self.time_warping(phase_features, time_coords)
            warped_embedding, warped_coords = self.phase_embedding(x, warped_times)
            phase_features = warped_embedding.abs().mean(dim=(-1, -2))  # [B, T]

            # Recompute phase_coords for downstream usage (take warped)
            phase_coords = warped_coords

        output = self.output_proj(phase_features)

        if return_phase_info:
            return output, {'phase_embedding': phase_embedding, 'phase_coords': phase_coords, 'phase_features': phase_features}
        return output


# -------------------------------
# FastQuantumBrain (FQB)
# -------------------------------
# Use the extended implementation with richer circuits and fitness functions.
from hybrid.quantum_brain_extended import FastQuantumBrain


# ---------------------------------------
# Hybrid Quantum-Temporal Network (HQTN)
# ---------------------------------------

class HybridQuantumTemporalNetwork:
    """
    Combines TPM (PyTorch) and FQB (Qiskit) into a single network.

    Pipeline:
    1) Run TPM on input sequence to get phase features and coordinates.
    2) Map TPM outputs to quantum circuit parameters in a structured way.
    3) Bind parameters and execute the circuit (Aer or IBM Runtime).
    4) Evaluate task-specific fitness.
    """

    def __init__(
        self,
        total_qubits: int,
        input_dim: int,
        manifold_dim: int = 64,
        num_frequencies: int = 8,
        use_time_warping: bool = True,
        task_type: str = 'entropy',
    ):
        self.tpm = TemporalPhaseManifold(
            input_dim=input_dim,
            output_dim=manifold_dim,  # we output manifold_dim features for richer control
            manifold_dim=manifold_dim,
            num_frequencies=num_frequencies,
            num_attention_layers=2,
            use_time_warping=use_time_warping,
        )
        self.qbrain = FastQuantumBrain(total_qubits=total_qubits, task_type=task_type)
        self.qc = self.qbrain.build_cognitive_circuit()

        # Region-aware parameter mapping:
        # For each brain region, create a dedicated mapper that outputs that region's parameter vector.
        # We identify parameter ownership by the Parameter.name prefix (e.g., "frontal_rot[...]", "amygdala_ent[...]").
        self.region_param_indices: Dict[str, List[int]] = {}
        self.region_param_counts: Dict[str, int] = {}

        # Build indices for parameters by region name
        for idx, p in enumerate(self.qbrain.all_parameters):
            # Parameter names are derived from ParameterVector, like 'frontal_rot0', 'frontal_ent3', etc.
            name = str(p)
            # Extract region prefix before the first underscore
            # e.g., "frontal_rot0" -> "frontal"
            region_prefix = name.split('_', 1)[0]
            self.region_param_indices.setdefault(region_prefix, []).append(idx)

        # Count per region
        for region_name, indices in self.region_param_indices.items():
            self.region_param_counts[region_name] = len(indices)

        # Create per-region mappers
        # Each mapper: [manifold_dim] -> [num_params_for_region]
        self.region_mappers = nn.ModuleDict()
        for region_name, count in self.region_param_counts.items():
            self.region_mappers[region_name] = nn.Sequential(
                nn.Linear(manifold_dim, manifold_dim),
                nn.ReLU(),
                nn.Linear(manifold_dim, count),
                nn.Sigmoid(),  # [0,1] -> later scaled to [0, 2π]
            )

    def _tpm_to_params(self, x: torch.Tensor) -> np.ndarray:
        """
        Aggregate TPM features across time and batch, then produce region-aware parameter vector.
        """
        with torch.no_grad():
            tpm_out, phase_info = self.tpm(x, return_phase_info=True)
            phase_features = phase_info['phase_features']  # [B, T, D] or [B, T]
            if phase_features.dim() == 2:
                phase_features = phase_features.unsqueeze(-1)  # [B, T, 1]

            # Aggregate across time and batch to a global control vector [D]
            aggregated = phase_features.mean(dim=1).mean(dim=0)  # [D]
            aggregated = aggregated.unsqueeze(0)  # [1, D]

            # Build full parameter vector by filling each region segment
            num_params = self.qbrain.num_params
            params_full = np.zeros((num_params,), dtype=np.float32)

            for region_name, indices in self.region_param_indices.items():
                mapper = self.region_mappers[region_name]
                # Generate [1, count] in [0,1]
                region_vals01 = mapper(aggregated).squeeze(0).cpu().numpy()
                # Scale to [0, 2π]
                region_vals = region_vals01 * (2 * np.pi)
                # Insert into the global param vector
                for local_idx, global_idx in enumerate(indices):
                    params_full[global_idx] = region_vals[local_idx]

            return params_full

    def quick_test(self, x: torch.Tensor, **task_kwargs) -> Tuple[Dict[str, int], float]:
        """
        Local Aer simulation to validate integration.
        """
        params = self._tpm_to_params(x)
        param_dict = {self.qbrain.all_parameters[i]: params[i] for i in range(len(params))}

        bound = self.qc.bind_parameters(param_dict)
        simulator = AerSimulator()
        transpiled = transpile(bound, simulator)
        job = simulator.run(transpiled, shots=1024)
        counts = job.result().get_counts()

        fitness = self.qbrain.compute_fitness(counts, **task_kwargs)

        return counts, fitness

    def run_on_ibm(
        self,
        x: torch.Tensor,
        backend_name: str = 'ibm_brisbane',
        num_iterations: int = 5,
        population_size: int = 4,
        shots: int = 512,
        **task_kwargs,
    ):
        """
        Hardware/runtime execution with simple evolutionary loop around TPM->params.
        Requires IBM Quantum account configured for QiskitRuntimeService.
        """
        if not HAS_IBM_RUNTIME:
            raise RuntimeError("qiskit-ibm-runtime not available. Install and configure IBM Quantum account.")

        service = QiskitRuntimeService(channel="ibm_quantum")
        backend = service.backend(backend_name)

        options = Options()
        options.execution.shots = shots
        options.optimization_level = 3
        options.resilience_level = 1

        # Initialize population by TPM-driven params + random perturbations
        base_params = self._tpm_to_params(x)
        rng = np.random.default_rng(0)
        population = [np.clip(base_params + rng.normal(0, 0.1, size=base_params.shape), 0, 2 * np.pi)
                      for _ in range(population_size)]
        fitness_history = []

        with Session(service=service, backend=backend) as session:
            sampler = Sampler(session=session, options=options)

            for it in range(num_iterations):
                circuits = []
                for params in population:
                    param_dict = {self.qbrain.all_parameters[i]: float(params[i]) for i in range(len(params))}
                    circuits.append(self.qc.bind_parameters(param_dict))

                job = sampler.run(circuits)
                result = job.result()

                fitness_scores = []
                for i in range(population_size):
                    quasi_dist = result.quasi_dists[i]
                    counts = {format(k, f'0{self.qbrain.total_qubits}b'): int(v * shots) for k, v in quasi_dist.items()}
                    fitness = self.qbrain.compute_fitness(counts, **task_kwargs)
                    fitness_scores.append(fitness)

                best_idx = int(np.argmax(fitness_scores))
                best_fitness = float(fitness_scores[best_idx])
                fitness_history.append(best_fitness)

                # Elitism + Gaussian mutations around best
                elite = population[best_idx].copy()
                new_population = [elite]
                for _ in range(population_size - 1):
                    mutant = np.clip(elite + rng.normal(0, 0.15, size=elite.shape), 0, 2 * np.pi)
                    new_population.append(mutant)
                population = new_population

        return population[0], fitness_history