from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RYGate, RZGate, CRYGate, CRZGate
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class NeuromodulationConfig:
  """Configuration for quantum neuromodulation system"""

  # Control qubit indices (emotional/interoceptive regions)
  amygdala_controls: List[int]
  insula_controls: List[int]
  frontal_controls: List[int]

  # Target qubit indices (regions to be modulated)
  frontal_targets: List[int]
  sensory_targets: List[int]
  emotional_targets: List[int]

  # Modulation strengths (rotation angles)
  fear_suppression_strength: float = np.pi / 4
  interoceptive_enhancement_strength: float = np.pi / 6
  executive_dampening_strength: float = np.pi / 8

  # Gate types for different modulations
  fear_gate_type: str = 'CRY'  # Controlled rotation for fear response
  interoceptive_gate_type: str = 'CRZ'  # Z-rotation for interoceptive enhancement
  executive_gate_type: str = 'CPhase'  # Phase gate for executive control


class StateDependentQuantumNeuromodulation:
  """
  State-Dependent Quantum Neuromodulation System

  This system implements quantum FiLM layers where:
  - Amygdala qubits control gates affecting frontal cortex (fear → executive suppression)
  - Insula qubits control gates affecting sensory regions (interoception → sensory enhancement)
  - Frontal qubits control gates affecting emotional regions (executive → emotional dampening)

  The key difference from classical NECA: modulation happens IN THE QUANTUM CIRCUIT,
  not as a post-processing step. The circuit's evolution is state-dependent.
  """

  def __init__(self, config: NeuromodulationConfig):
    self.config = config
    self.modulation_history = []

  def add_fear_suppression_layer(self, qc: QuantumCircuit, modulation_strength: Parameter = None):
    """
    Add state-dependent fear suppression gates.

    Biological Analogue: High amygdala activation → suppress frontal executive function
    Quantum Implementation: Amygdala qubits control CRY gates on frontal qubits

    When amygdala qubits are in |1⟩ (high activation), the controlled rotation
    is applied, suppressing the frontal qubit's contribution to the final state.
    """
    strength = modulation_strength or self.config.fear_suppression_strength

    # Each amygdala control qubit modulates multiple frontal targets
    for control_idx in self.config.amygdala_controls:
      for target_idx in self.config.frontal_targets:
        if self.config.fear_gate_type == 'CRY':
          qc.cry(-strength, control_idx, target_idx)
        elif self.config.fear_gate_type == 'CRZ':
          qc.crz(-strength, control_idx, target_idx)
        elif self.config.fear_gate_type == 'CPhase':
          qc.cp(-strength, control_idx, target_idx)

    return qc

  def add_interoceptive_enhancement_layer(self, qc: QuantumCircuit, modulation_strength: Parameter = None):
    """
    Add state-dependent interoceptive enhancement gates.

    Biological Analogue: High insula activation → enhance sensory processing
    Quantum Implementation: Insula qubits control CRZ gates on sensory qubits

    When insula qubits are in |1⟩ (high interoceptive awareness), the controlled
    rotation enhances the sensory qubit's phase, amplifying its signal.
    """
    strength = modulation_strength or self.config.interoceptive_enhancement_strength

    for control_idx in self.config.insula_controls:
      for target_idx in self.config.sensory_targets:
        if self.config.interoceptive_gate_type == 'CRZ':
          qc.crz(strength, control_idx, target_idx)
        elif self.config.interoceptive_gate_type == 'CRY':
          qc.cry(strength, control_idx, target_idx)
        elif self.config.interoceptive_gate_type == 'CPhase':
          qc.cp(strength, control_idx, target_idx)

    return qc

  def add_executive_dampening_layer(self, qc: QuantumCircuit, modulation_strength: Parameter = None):
    """
    Add state-dependent executive control dampening.

    Biological Analogue: Strong executive function → dampen emotional reactivity
    Quantum Implementation: Frontal qubits control gates affecting emotional regions

    When frontal qubits are in |1⟩ (high executive control), emotional qubits
    are dampened through controlled phase shifts or rotations.
    """
    strength = modulation_strength or self.config.executive_dampening_strength

    for control_idx in self.config.frontal_controls:
      for target_idx in self.config.emotional_targets:
        if self.config.executive_gate_type == 'CPhase':
          qc.cp(-strength, control_idx, target_idx)
        elif self.config.executive_gate_type == 'CRY':
          qc.cry(-strength, control_idx, target_idx)
        elif self.config.executive_gate_type == 'CRZ':
          qc.crz(-strength, control_idx, target_idx)

    return qc

  def add_bidirectional_feedback(self, qc: QuantumCircuit):
    """
    Add bidirectional feedback loops between regions.

    This creates a quantum feedback network where:
    - Amygdala ⟷ Frontal (fear ⟷ executive control)
    - Insula ⟷ Sensory (interoception ⟷ perception)
    - Frontal ⟷ Emotional (cognition ⟷ affect)

    Biological Analogue: Reciprocal connections in brain networks
    """

    # Amygdala → Frontal (fear suppresses executive)
    self.add_fear_suppression_layer(qc)

    # Frontal → Amygdala (executive controls fear)
    for control_idx in self.config.frontal_controls[:min(3, len(self.config.frontal_controls))]:
      for target_idx in self.config.amygdala_controls:
        qc.cry(-self.config.executive_dampening_strength / 2, control_idx, target_idx)

    # Insula → Sensory (interoception enhances sensing)
    self.add_interoceptive_enhancement_layer(qc)

    # Sensory → Insula (sensory input affects body awareness)
    for control_idx in self.config.sensory_targets[:min(3, len(self.config.sensory_targets))]:
      for target_idx in self.config.insula_controls:
        qc.crz(self.config.interoceptive_enhancement_strength / 2, control_idx, target_idx)

    return qc

  def add_parameterized_modulation(self, qc: QuantumCircuit, param_vector: ParameterVector):
    """
    Add learnable parameterized neuromodulation gates.

    This allows the modulation strengths to be optimized during training,
    learning the optimal balance between different neuromodulatory pathways.
    """
    param_idx = 0

    # Parameterized fear suppression
    for control_idx in self.config.amygdala_controls:
      for target_idx in self.config.frontal_targets[:3]:  # Limit to first 3 to save params
        if param_idx < len(param_vector):
          qc.cry(param_vector[param_idx], control_idx, target_idx)
          param_idx += 1

    # Parameterized interoceptive enhancement
    for control_idx in self.config.insula_controls:
      for target_idx in self.config.sensory_targets[:3]:
        if param_idx < len(param_vector):
          qc.crz(param_vector[param_idx], control_idx, target_idx)
          param_idx += 1

    # Parameterized executive dampening
    for control_idx in self.config.frontal_controls[:2]:
      for target_idx in self.config.emotional_targets:
        if param_idx < len(param_vector):
          qc.cp(param_vector[param_idx], control_idx, target_idx)
          param_idx += 1

    return qc, param_idx

  def add_multi_qubit_modulation(self, qc: QuantumCircuit):
    """
    Add multi-qubit control gates for complex neuromodulation.

    Uses Toffoli-like gates where multiple emotional/cognitive states
    must be active simultaneously to trigger modulation.

    Example: High amygdala AND low frontal → strong fear response
    """
    # If we have enough qubits, add 3-qubit gates
    if (len(self.config.amygdala_controls) >= 2 and len(self.config.frontal_targets) >= 1):
      # Two amygdala qubits control one frontal target
      # Models: combined threat signals → executive suppression
      amyg1 = self.config.amygdala_controls[0]
      amyg2 = self.config.amygdala_controls[1]
      frontal = self.config.frontal_targets[0]

      # Decompose Toffoli into 2-qubit gates (placeholder: use ccx directly if available)
      qc.ccx(amyg1, amyg2, frontal)

    return qc

  def add_adaptive_modulation_layer(self, qc: QuantumCircuit, arousal_threshold: float = 0.5):
    """
    Add modulation that adapts based on overall system arousal.

    Uses ancilla qubit to compute overall arousal level, then uses it
    to control the strength of neuromodulation across the entire system.
    """
    # Placeholder for advanced implementation requiring ancilla and arithmetic
    return qc


class QuantumFiLMLayer:
  """
  Direct quantum analogue of classical FiLM (Feature-wise Linear Modulation) layer.

  Classical FiLM: output = γ ⊙ features + β
  Quantum FiLM: |ψ_out⟩ = U_modulation(γ, β) |ψ_features⟩

  Where U_modulation is a parameterized unitary controlled by "emotional" qubits.
  """

  def __init__(self, control_qubits: List[int], target_qubits: List[int]):
    self.control_qubits = control_qubits
    self.target_qubits = target_qubits

  def apply_gamma_modulation(self, qc: QuantumCircuit, gamma_param: Parameter):
    """
    Apply γ (gain) modulation via controlled rotations.

    When control qubits are in |1⟩ (high activation), target qubits
    are rotated by γ, amplifying or suppressing their contribution.
    """
    for control in self.control_qubits:
      for target in self.target_qubits:
        qc.cry(gamma_param, control, target)
    return qc

  def apply_beta_modulation(self, qc: QuantumCircuit, beta_param: Parameter):
    """
    Apply β (bias) modulation via controlled phase shifts.

    When control qubits are in |1⟩, target qubits receive a phase shift β,
    equivalent to adding a bias term in classical networks.
    """
    for control in self.control_qubits:
      for target in self.target_qubits:
        qc.cp(beta_param, control, target)
    return qc

  def apply_full_film(self, qc: QuantumCircuit, gamma_param: Parameter, beta_param: Parameter):
    """Apply complete quantum FiLM transformation"""
    self.apply_gamma_modulation(qc, gamma_param)
    self.apply_beta_modulation(qc, beta_param)
    return qc


def create_neuromodulated_circuit(total_qubits: int = 133) -> Tuple[QuantumCircuit, NeuromodulationConfig]:
  """
  Create a complete quantum cognitive circuit with state-dependent neuromodulation.

  This implements the full NECA architecture with quantum neuromodulation layers
  inserted at strategic points in the information flow.

  Returns:
    tuple: (circuit, neuromodulation_config)
  """
  # Define brain regions
  regions = {
    'thalamus': list(range(0, 8)),
    'occipital': list(range(8, 28)),
    'parietal': list(range(28, 48)),
    'temporal': list(range(48, 73)),
    'hippocampus': list(range(73, 88)),
    'amygdala': list(range(88, 96)),
    'frontal': list(range(96, 121)),
    'cerebellum': list(range(121, 133)),
  }

  # Create neuromodulation configuration
  config = NeuromodulationConfig(
    amygdala_controls=regions['amygdala'][:4],  # First 4 amygdala qubits as controls
    insula_controls=regions['hippocampus'][:4],  # Use hippocampus as insula proxy
    frontal_controls=regions['frontal'][:4],   # First 4 frontal qubits

    frontal_targets=regions['frontal'][4:12],  # Middle frontal qubits
    sensory_targets=regions['occipital'][:8] + regions['parietal'][:4],
    emotional_targets=regions['amygdala'][4:],  # Remaining amygdala qubits

    fear_suppression_strength=np.pi / 4,
    interoceptive_enhancement_strength=np.pi / 6,
    executive_dampening_strength=np.pi / 8,
  )

  # Initialize circuit
  qr = QuantumRegister(total_qubits, 'brain')
  cr = ClassicalRegister(total_qubits, 'measure')
  qc = QuantumCircuit(qr, cr)

  # Initialize neuromodulation system
  neuromod = StateDependentQuantumNeuromodulation(config)

  # Phase 1: Input Encoding
  for q in regions['thalamus']:
    qc.h(q)

  # Basic sensory processing
  for i in range(len(regions['occipital'])):
    qc.ry(np.pi / 4, regions['occipital'][i])

  # Phase 2: Reactive Pathway (Low Road)
  for i in range(min(4, len(regions['thalamus']), len(regions['amygdala']))):
    qc.cx(regions['thalamus'][i], regions['amygdala'][i])
  for q in regions['amygdala']:
    qc.ry(np.pi / 6, q)

  # Phase 3: First Neuromodulation Layer (Fear Response)
  neuromod.add_fear_suppression_layer(qc)

  # Phase 4: Deliberative Pathway (High Road) - State-Dependent
  for i in range(min(10, len(regions['occipital']), len(regions['temporal']))):
    qc.cx(regions['occipital'][i], regions['temporal'][i])
  for q in regions['hippocampus']:
    qc.ry(np.pi / 5, q)

  # Phase 5: Second Neuromodulation Layer (Interoceptive Enhancement)
  neuromod.add_interoceptive_enhancement_layer(qc)

  # Phase 6: Executive Function (Modulated by Emotion & Interoception)
  for i in range(min(8, len(regions['hippocampus']), len(regions['frontal']))):
    qc.cx(regions['hippocampus'][i], regions['frontal'][i])
  for i in range(min(6, len(regions['amygdala']), len(regions['frontal']))):
    qc.cx(regions['amygdala'][i], regions['frontal'][i + 8])
  for q in regions['frontal']:
    qc.ry(np.pi / 3, q)

  # Phase 7: Third Neuromodulation Layer (Executive Control)
  neuromod.add_executive_dampening_layer(qc)

  # Phase 8: Bidirectional Feedback Loops
  neuromod.add_bidirectional_feedback(qc)

  # Phase 9: Motor Output (Cerebellum)
  for i in range(min(8, len(regions['frontal']), len(regions['cerebellum']))):
    qc.cx(regions['frontal'][i], regions['cerebellum'][i])
  for q in regions['cerebellum']:
    qc.ry(np.pi / 4, q)

  # Measurement
  qc.measure(qr, cr)

  return qc, config


def demonstrate_quantum_film_layer() -> QuantumCircuit:
  """Demonstrate isolated quantum FiLM layer"""
  qc = QuantumCircuit(8, 8)

  # Control qubits (emotional state)
  control_qubits = [0, 1]

  # Target qubits (features to modulate)
  target_qubits = [4, 5, 6, 7]

  # Initialize control qubits in superposition
  for q in control_qubits:
    qc.h(q)

  # Initialize target qubits with some state
  for q in target_qubits:
    qc.ry(np.pi / 4, q)

  film = QuantumFiLMLayer(control_qubits, target_qubits)

  # Define modulation parameters
  gamma = Parameter('gamma')
  beta = Parameter('beta')

  # Apply quantum FiLM
  film.apply_full_film(qc, gamma, beta)

  # Measure
  qc.measure_all()

  return qc


class EnhancedQuantumBrain:
  """
  Enhanced Quantum Brain with State-Dependent Neuromodulation.
  Drop-in replacement for FastQuantumBrain with quantum FiLM layers.
  """

  def __init__(self, total_qubits=133, task_type='homeostatic', use_neca=True, use_quantum_neuromod=True):
    self.total_qubits = total_qubits
    self.task_type = task_type
    self.use_neca = use_neca
    self.use_quantum_neuromod = use_quantum_neuromod

    # Store circuit and config
    self.circuit_template = None
    self.neuromod_config = None
    self.neuromod_system = None

  def build_circuit(self):
    """Build circuit with optional quantum neuromodulation"""
    if self.use_quantum_neuromod:
      self.circuit_template, self.neuromod_config = create_neuromodulated_circuit(self.total_qubits)
      self.neuromod_system = StateDependentQuantumNeuromodulation(self.neuromod_config)
    else:
      # Use standard circuit building
      self.circuit_template = self._build_standard_circuit()

    return self.circuit_template

  def _build_standard_circuit(self):
    """Fallback to standard circuit without quantum neuromodulation"""
    qr = QuantumRegister(self.total_qubits, 'brain')
    cr = ClassicalRegister(self.total_qubits, 'measure')
    qc = QuantumCircuit(qr, cr)

    # Basic cognitive circuit (simplified)
    for i in range(self.total_qubits):
      qc.h(i)
      if i < self.total_qubits - 1:
        qc.cx(i, i + 1)

    qc.measure_all()
    return qc

  def analyze_neuromodulation_impact(self, counts: Dict) -> Dict:
    """
    Analyze how neuromodulation affected the circuit's behavior.
    Returns metrics showing:
    - Emotional activation levels
    - Cognitive control strength
    - Modulation effectiveness
    """
    if not self.use_quantum_neuromod:
      return {}

    total = sum(counts.values())

    # Extract activation patterns from control regions
    amygdala_activation = self._compute_region_activation(counts, self.neuromod_config.amygdala_controls, total)
    insula_activation = self._compute_region_activation(counts, self.neuromod_config.insula_controls, total)
    frontal_activation = self._compute_region_activation(counts, self.neuromod_config.frontal_controls, total)

    # Compute modulation strength (how much control affected targets)
    frontal_target_activation = self._compute_region_activation(counts, self.neuromod_config.frontal_targets, total)
    sensory_activation = self._compute_region_activation(counts, self.neuromod_config.sensory_targets, total)

    # Calculate effective modulation
    fear_modulation_strength = amygdala_activation * (1 - frontal_target_activation)
    intero_modulation_strength = insula_activation * sensory_activation
    executive_modulation_strength = frontal_activation * (1 - amygdala_activation)

    return {
      'amygdala_activation': amygdala_activation,
      'insula_activation': insula_activation,
      'frontal_activation': frontal_activation,
      'fear_modulation': fear_modulation_strength,
      'interoceptive_modulation': intero_modulation_strength,
      'executive_modulation': executive_modulation_strength,
      'overall_neuromod_activity': (fear_modulation_strength + intero_modulation_strength + executive_modulation_strength) / 3,
    }

  def _compute_region_activation(self, counts: Dict, qubit_indices: List[int], total: int) -> float:
    """Compute average activation (proportion of |1⟩) for a region"""
    activation = 0.0
    for bitstring, count in counts.items():
      prob = count / total
      ones_in_region = sum(1 for idx in qubit_indices if bitstring[idx] == '1')
      region_activation = ones_in_region / len(qubit_indices)
      activation += prob * region_activation
    return activation


def _compute_entropy(counts: Dict, shots: int) -> float:
  """Compute Shannon entropy of measurement distribution"""
  entropy = 0.0
  for count in counts.values():
    prob = count / shots
    if prob > 0:
      entropy -= prob * np.log2(prob)
  return entropy