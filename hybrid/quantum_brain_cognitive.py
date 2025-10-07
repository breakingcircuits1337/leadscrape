from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes, EfficientSU2, TwoLocal
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Options
from qiskit.primitives import BackendEstimator
import numpy as np
from collections import defaultdict
import time


class BrainRegion:
  """Structured representation of a brain region with named parameters"""

  def __init__(self, name, qubits, function, depth=2):
    self.name = name
    self.qubits = qubits
    self.function = function
    self.depth = depth

    # Create structured parameter vectors for this region
    num_qubits = len(qubits)
    self.params = {
      'rotation': ParameterVector(f'{name}_rot', num_qubits * depth),
      'entanglement': ParameterVector(f'{name}_ent', max(0, (num_qubits - 1) * depth))
    }

  def get_all_params(self):
    """Returns all parameters as a flat list"""
    all_params = []
    for param_vec in self.params.values():
      all_params.extend(param_vec)
    return all_params

  def param_count(self):
    """Returns total number of parameters"""
    return sum(len(pv) for pv in self.params.values())


class CognitiveFitnessEvaluator:
  """Task-specific fitness functions for cognitive evaluation"""

  @staticmethod
  def entropy_fitness(counts):
    """Default: Shannon entropy (measures quantum complexity)"""
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
    max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1
    return entropy / max_entropy if max_entropy > 0 else 0

  @staticmethod
  def pattern_matching_fitness(counts, target_pattern):
    """Task: Match specific output pattern"""
    total = sum(counts.values())
    target_count = counts.get(target_pattern, 0)
    return target_count / total

  @staticmethod
  def decision_making_fitness(counts, num_choices=4):
    """
    Task: Decision-making
    Rewards clear, decisive outputs (low entropy within top choices)
    """
    total = sum(counts.values())
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # Get top N choices
    top_choices = sorted_counts[:num_choices]
    top_total = sum(c for _, c in top_choices)

    # High fitness if most probability mass is in top choices
    decisiveness = top_total / total

    # Among top choices, prefer clear winner (low entropy)
    if top_total > 0:
      top_probs = [c / top_total for _, c in top_choices]
      top_entropy = -sum(p * np.log2(p + 1e-10) for p in top_probs if p > 0)
      max_top_entropy = np.log2(num_choices)
      clarity = 1 - (top_entropy / max_top_entropy)
    else:
      clarity = 0

    # Combined score
    return 0.7 * decisiveness + 0.3 * clarity

  @staticmethod
  def memory_retrieval_fitness(counts, memory_patterns):
    """
    Task: Memory retrieval
    Rewards outputs that match any stored memory pattern
    """
    total = sum(counts.values())
    memory_matches = sum(counts.get(pattern, 0) for pattern in memory_patterns)
    return memory_matches / total

  @staticmethod
  def classification_fitness(counts, class_patterns):
    """
    Task: Classification into categories
    class_patterns = {'class_A': [...patterns...], 'class_B': [...patterns...]}
    Rewards concentration into one class
    """
    total = sum(counts.values())

    class_scores = {}
    for class_name, patterns in class_patterns.items():
      class_count = sum(counts.get(p, 0) for p in patterns)
      class_scores[class_name] = class_count / total

    # Reward maximum class membership
    return max(class_scores.values()) if class_scores else 0

  @staticmethod
  def spatial_reasoning_fitness(counts):
    """
    Task: Spatial reasoning
    Rewards outputs with structured patterns (e.g., symmetry, locality)
    """
    total = sum(counts.values())

    # Analyze bit patterns for spatial structure
    symmetry_score = 0
    locality_score = 0

    for bitstring, count in counts.items():
      prob = count / total

      # Check for symmetry (palindromic patterns)
      if bitstring == bitstring[::-1]:
        symmetry_score += prob

      # Check for locality (consecutive 1s or 0s)
      runs = 0
      current_run = 1
      for i in range(1, len(bitstring)):
        if bitstring[i] == bitstring[i-1]:
          current_run += 1
        else:
          runs += 1
          current_run = 1

      # Longer runs = more local structure
      avg_run_length = len(bitstring) / (runs + 1) if runs > 0 else len(bitstring)
      locality_score += prob * (avg_run_length / len(bitstring))

    return 0.5 * symmetry_score + 0.5 * locality_score

  @staticmethod
  def working_memory_fitness(counts, capacity=7):
    """
    Task: Working memory (hold multiple items)
    Rewards maintaining diverse but limited set of active states
    Inspired by Miller's "magical number 7±2"
    """
    total = sum(counts.values())
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # Get top 'capacity' states
    active_states = sorted_counts[:capacity]
    active_total = sum(c for _, c in active_states)

    # Reward high concentration in working memory slots
    concentration = active_total / total

    # Reward even distribution among active states (all slots used equally)
    if active_total > 0:
      active_probs = [c / active_total for _, c in active_states]
      uniformity = 1 - np.std(active_probs) / (np.mean(active_probs) + 1e-10)
    else:
      uniformity = 0

    return 0.6 * concentration + 0.4 * uniformity

  @staticmethod
  def attention_focus_fitness(counts, focus_region_size=8):
    """
    Task: Attention/Focus
    Rewards concentration on specific qubit region while suppressing others
    Models selective attention by rewarding localized activation
    """
    total = sum(counts.values())

    # Analyze which qubits are most active
    num_qubits = len(list(counts.keys())[0]) if counts else 0
    qubit_activity = [0] * num_qubits

    for bitstring, count in counts.items():
      prob = count / total
      for i, bit in enumerate(bitstring):
        if bit == '1':
          qubit_activity[i] += prob

    # Find most active region
    if num_qubits >= focus_region_size:
      max_regional_activity = 0
      for start in range(num_qubits - focus_region_size + 1):
        region_activity = sum(qubit_activity[start:start + focus_region_size])
        max_regional_activity = max(max_regional_activity, region_activity)
      # Normalize by perfect focus
      focus_score = max_regional_activity / focus_region_size
    else:
      focus_score = sum(qubit_activity) / num_qubits

    return focus_score

  @staticmethod
  def sequence_learning_fitness(counts, target_sequence):
    """
    Task: Sequence learning
    Rewards patterns that follow a temporal sequence
    target_sequence = ['state1', 'state2', 'state3']
    Measures how well outputs progress through sequence
    """
    total = sum(counts.values())

    # Score based on sequence position matching
    sequence_score = 0
    for i, target_state in enumerate(target_sequence):
      # Weight later sequence elements higher (goal state)
      position_weight = (i + 1) / len(target_sequence)
      state_prob = counts.get(target_state, 0) / total
      sequence_score += position_weight * state_prob

    return sequence_score

  @staticmethod
  def problem_solving_fitness(counts, initial_state, goal_state, forbidden_states=None):
    """
    Task: Problem solving (state space search)
    Rewards finding path from initial to goal while avoiding forbidden states
    """
    if forbidden_states is None:
      forbidden_states = set()

    total = sum(counts.values())

    # Reward reaching goal state
    goal_prob = counts.get(goal_state, 0) / total

    # Penalize forbidden states
    forbidden_prob = sum(counts.get(fs, 0) for fs in forbidden_states) / total

    # Reward intermediate states (not initial, not forbidden)
    exploration_states = {k for k in counts.keys() 
              if k != initial_state and k not in forbidden_states and k != goal_state}
    exploration_prob = sum(counts.get(s, 0) for s in exploration_states) / total

    # Combined score
    return 0.6 * goal_prob + 0.3 * exploration_prob - 0.1 * forbidden_prob

  @staticmethod
  def cognitive_flexibility_fitness(counts, num_contexts=4):
    """
    Task: Cognitive flexibility (task switching)
    Rewards ability to produce multiple distinct response modes
    Measures adaptability by rewarding multi-modal distributions
    """
    total = sum(counts.values())
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # Find distinct modes (local maxima in probability space)
    modes = []
    for i, (state, count) in enumerate(sorted_counts[:num_contexts * 2]):
      prob = count / total
      # Check if this is sufficiently different from existing modes
      is_distinct = all(
        sum(c1 != c2 for c1, c2 in zip(state, mode)) >= len(state) * 0.3
        for mode in modes
      )
      if is_distinct:
        modes.append(state)
        if len(modes) >= num_contexts:
          break

    # Reward number of distinct modes found
    mode_diversity = len(modes) / num_contexts

    # Reward balanced probability across modes
    if modes:
      mode_probs = [counts.get(m, 0) / total for m in modes]
      balance = 1 - np.std(mode_probs) / (np.mean(mode_probs) + 1e-10)
    else:
      balance = 0

    return 0.6 * mode_diversity + 0.4 * balance

  @staticmethod
  def error_detection_fitness(counts, correct_patterns, error_patterns):
    """
    Task: Error detection and correction
    Rewards avoiding error states and converging to correct states
    """
    total = sum(counts.values())

    # Reward correct patterns
    correct_prob = sum(counts.get(p, 0) for p in correct_patterns) / total

    # Penalize error patterns
    error_prob = sum(counts.get(p, 0) for p in error_patterns) / total

    # Reward concentration (not spreading across many states)
    top_10_prob = sum(c for _, c in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]) / total

    return 0.5 * correct_prob - 0.3 * error_prob + 0.2 * top_10_prob

  @staticmethod
  def inhibition_control_fitness(counts, target_pattern, distractor_patterns):
    """
    Task: Inhibition control (resist distractors)
    Rewards selecting target while suppressing distractors
    Models impulse control and interference resistance
    """
    total = sum(counts.values())

    # Reward target selection
    target_prob = counts.get(target_pattern, 0) / total

    # Penalize distractor selection
    distractor_prob = sum(counts.get(d, 0) for d in distractor_patterns) / total

    # Reward decisiveness (strong target, weak distractors)
    if distractor_prob > 0:
      inhibition_ratio = target_prob / (distractor_prob + 1e-10)
      inhibition_score = min(1.0, inhibition_ratio / 5.0)  # Normalize
    else:
      inhibition_score = 1.0 if target_prob > 0.5 else 0.0

    return 0.5 * target_prob + 0.5 * inhibition_score

  @staticmethod
  def analogical_reasoning_fitness(counts, source_pattern, target_pattern, relation_distance=5):
    """
    Task: Analogical reasoning
    "A is to B as C is to ?"
    Rewards finding patterns with same structural relationship
    """
    total = sum(counts.values())

    # Compute relationship in source (A to B)
    if len(source_pattern) >= 2:
      source_relation = [int(source_pattern[0][i] != source_pattern[1][i]) 
               for i in range(len(source_pattern[0]))]
    else:
      return 0

    # Find outputs that apply same relationship to target
    analogy_score = 0
    for output_state, count in counts.items():
      prob = count / total

      # Check if output relates to target_pattern like source relates
      if len(target_pattern) > 0:
        output_relation = [int(target_pattern[0][i] != output_state[i]) 
                 for i in range(min(len(target_pattern[0]), len(output_state)))]

        # Measure relationship similarity
        matching_bits = sum(1 for i in range(min(len(source_relation), len(output_relation)))
                 if source_relation[i] == output_relation[i])
        similarity = matching_bits / len(source_relation)

        analogy_score += prob * similarity

    return analogy_score

  @staticmethod
  def reward_learning_fitness(counts, reward_states, penalty_states, reward_weight=1.0, penalty_weight=0.5):
    """
    Task: Reward-based learning
    Rewards converging to high-reward states and avoiding penalties
    Models reinforcement learning behavior
    """
    total = sum(counts.values())

    # Calculate reward
    reward_prob = sum(counts.get(rs, 0) for rs in reward_states) / total
    penalty_prob = sum(counts.get(ps, 0) for ps in penalty_states) / total

    # Net reward
    net_reward = reward_weight * reward_prob - penalty_weight * penalty_prob

    # Bonus for strong preference (decisive learning)
    decisiveness_bonus = 0.2 if reward_prob > 0.5 else 0

    return max(0, net_reward + decisiveness_bonus)

  @staticmethod
  def predictive_coding_fitness(counts, expected_pattern, prediction_error_threshold=0.3):
    """
    Task: Predictive coding
    Rewards outputs that match expectations (low prediction error)
    or signal large prediction errors for unexpected inputs
    """
    total = sum(counts.values())

    # Measure prediction error for each output
    error_distribution = []
    for output_state, count in counts.items():
      prob = count / total

      # Hamming distance as prediction error
      if len(output_state) == len(expected_pattern):
        error = sum(c1 != c2 for c1, c2 in zip(output_state, expected_pattern)) / len(output_state)
        error_distribution.append((error, prob))

    # Reward bimodal distribution: either very accurate or clear error signal
    low_error_prob = sum(prob for error, prob in error_distribution if error < prediction_error_threshold)
    high_error_prob = sum(prob for error, prob in error_distribution if error > (1 - prediction_error_threshold))

    clarity = low_error_prob + high_error_prob

    return clarity


class FastQuantumBrain:
  """High-performance quantum cognitive system using 133 qubits"""

  def __init__(self, total_qubits=133, task_type='entropy'):
    """High-performance quantum cognitive system with a Trainable Time Manifold."""
    self.total_qubits = total_qubits
    self.task_type = task_type

    # Optimized qubit allocation with a dedicated Temporal Control region
    self.regions = {
      'temporal_control': BrainRegion('temporal_control', list(range(0, 5)), 'dynamic_time_warping', depth=1),
      'thalamus':     BrainRegion('thalamus', list(range(5, 13)), 'input_routing', depth=1),
      'occipital':    BrainRegion('occipital', list(range(13, 33)), 'visual_processing', depth=2),
      'parietal':     BrainRegion('parietal', list(range(33, 53)), 'spatial_integration', depth=2),
      'temporal':     BrainRegion('temporal', list(range(53, 78)), 'memory_processing', depth=2),
      'hippocampus':  BrainRegion('hippocampus', list(range(78, 93)), 'memory_formation', depth=2),
      'amygdala':     BrainRegion('amygdala', list(range(93, 101)), 'emotional_tagging', depth=1),
      'frontal':      BrainRegion('frontal', list(range(101, 121)), 'decision_making', depth=3),
      'cerebellum':   BrainRegion('cerebellum', list(range(121, 133)), 'motor_control', depth=1),
    }

    # Build circuit and get exact parameter count
    self.circuit_template = None
    self.all_parameters = None
    self.num_params = 0

    # Chaos parameters
    self.chaos_r = 3.99
    self.chaos_seed = 0.5
    self.mutation_strength = 0.2

    # Adaptive mutation
    self.min_mutation_strength = 0.05
    self.max_mutation_strength = 0.5
    self.plateau_threshold = 3
    self.plateau_counter = 0
    self.best_fitness_ever = 0.0

    # Multi-attractor chaos
    self.chaos_attractors = [3.99, 3.85, 3.70, 3.57]
    self.current_attractor_idx = 0

    # Tracking
    self.lyapunov_history = []
    self.chaos_history = []

    # Fitness evaluator
    self.fitness_evaluator = CognitiveFitnessEvaluator()

  def _add_region_layer(self, qc, region, param_offset):
    """
    Adds a processing layer for a specific brain region
    Returns updated parameter offset
    """
    qubits = region.qubits
    rot_params = region.params['rotation']
    ent_params = region.params['entanglement']

    rot_idx = 0
    ent_idx = 0

    for layer in range(region.depth):
      # Rotation layer
      for i, q in enumerate(qubits):
        qc.ry(rot_params[rot_idx], q)
        rot_idx += 1

      # Entanglement layer
      for i in range(len(qubits) - 1):
        qc.cx(qubits[i], qubits[i + 1])
        if ent_idx < len(ent_params):
          qc.rz(ent_params[ent_idx], qubits[i + 1])
          ent_idx += 1

    return param_offset + region.param_count()

  def _add_temporally_controlled_layer(self, qc, processing_qubits, control_qubits, theta, tau):
    """
    Applies a processing layer where the evolution is scaled by a time parameter 'tau'
    and controlled by the state of 'control_qubits'.
    """
    num_p = len(processing_qubits)
    clock_qubit = control_qubits[0]

    # Controlled rotations with angle theta * tau
    for i in range(num_p):
      qc.cry(theta[i] * tau[i], clock_qubit, processing_qubits[i])

    # Controlled entanglement using second control qubit if available
    if len(control_qubits) > 1:
      for i in range(num_p - 1):
        qc.ccx(control_qubits[1], processing_qubits[i], processing_qubits[i + 1])

  def build_cognitive_circuit(self):
    """
    Builds the complete cognitive circuit with structured parameters
    Parameters are automatically tracked and counted
    """
    qr = QuantumRegister(self.total_qubits, 'brain')
    cr = ClassicalRegister(self.total_qubits, 'measure')
    qc = QuantumCircuit(qr, cr)

    # PHASE 1: Input encoding (Thalamus)
    thal = self.regions['thalamus']
    for q in thal.qubits:
      qc.h(q)
    self._add_region_layer(qc, thal, 0)

    # PHASE 2: Sensory processing (Occipital & Parietal in parallel)
    occ = self.regions['occipital']
    par = self.regions['parietal']

    # Connect thalamus to sensory regions
    for i, tq in enumerate(thal.qubits[:4]):
      if i < len(occ.qubits):
        qc.cx(tq, occ.qubits[i])
      if i < len(par.qubits):
        qc.cx(tq, par.qubits[i])

    # Process in parallel
    self._add_region_layer(qc, occ, 0)
    self._add_region_layer(qc, par, 0)

    # PHASE 3: Memory system (Temporal + Hippocampus + Amygdala)
    temp = self.regions['temporal']
    hippo = self.regions['hippocampus']
    amyg = self.regions['amygdala']

    # Feed sensory to temporal
    for i in range(min(10, len(occ.qubits), len(temp.qubits))):
      qc.cx(occ.qubits[i], temp.qubits[i])

    self._add_region_layer(qc, temp, 0)

    # Memory formation
    for i in range(min(len(temp.qubits), len(hippo.qubits))):
      qc.cx(temp.qubits[i], hippo.qubits[i])

    self._add_region_layer(qc, hippo, 0)

    # Emotional tagging
    for i in range(min(len(hippo.qubits), len(amyg.qubits))):
      qc.cx(hippo.qubits[i], amyg.qubits[i])

    self._add_region_layer(qc, amyg, 0)

    # PHASE 4: Executive function (Frontal)
    front = self.regions['frontal']

    # Integrate information
    for i in range(min(len(hippo.qubits), len(front.qubits))):
      qc.cx(hippo.qubits[i], front.qubits[i])

    for i in range(min(len(amyg.qubits), len(front.qubits))):
      qc.cx(amyg.qubits[i], front.qubits[i + 8])

    self._add_region_layer(qc, front, 0)

    # PHASE 5: Motor control (Cerebellum)
    cereb = self.regions['cerebellum']

    for i in range(min(8, len(front.qubits), len(cereb.qubits))):
      qc.cx(front.qubits[i], cereb.qubits[i])

    self._add_region_layer(qc, cereb, 0)

    # Measurement
    qc.measure(qr, cr)

    # Store circuit and parameters
    self.circuit_template = qc
    self.all_parameters = list(qc.parameters)
    self.num_params = len(self.all_parameters)

    return qc

  def build_fast_cognitive_circuit(self, depth=2):
    """
    Builds a cognitive circuit with a data-driven, non-linear time manifold.
    """
    qr = QuantumRegister(self.total_qubits, 'brain')
    cr = ClassicalRegister(self.total_qubits, 'measure')
    qc = QuantumCircuit(qr, cr)

    # Calculate the number of parameters needed
    param_count = 0
    processing_regions = {k: v for k, v in self.regions.items() if k != 'temporal_control'}
    for region in processing_regions.values():
      param_count += len(region.qubits) * depth
    param_count += len(self.regions['temporal_control'].qubits)

    # Create parameter vectors
    theta = ParameterVector('theta', param_count)
    tau = ParameterVector('tau', param_count)
    self.num_params = param_count * 2

    p_idx = 0

    # PHASE 1: TEMPORAL & SENSORY ENCODING
    temp_q = self.regions['temporal_control'].qubits
    thal_q = self.regions['thalamus'].qubits

    for q in thal_q + temp_q:
      qc.h(q)

    for i, q in enumerate(thal_q):
      qc.rz(theta[p_idx], q)
      p_idx += 1

    for i in range(min(len(thal_q), len(temp_q))):
      qc.cx(thal_q[i], temp_q[i])

    for i, q in enumerate(temp_q):
      qc.ry(theta[p_idx], q)
      p_idx += 1
    for i in range(len(temp_q) - 1):
      qc.cz(temp_q[i], temp_q[i + 1])

    qc.barrier()

    # PHASE 2: TEMPORALLY-CONTROLLED COGNITION
    for region_name in ['occipital', 'parietal', 'temporal', 'frontal']:
      region_qubits = self.regions[region_name].qubits
      for layer in range(depth):
        theta_slice = theta[p_idx: p_idx + len(region_qubits)]
        tau_slice = tau[p_idx: p_idx + len(region_qubits)]
        self._add_temporally_controlled_layer(qc, region_qubits, temp_q, theta_slice, tau_slice)
        p_idx += len(region_qubits)
      qc.barrier()

    # PHASE 3: FINAL INTEGRATION & ACTION
    frontal_q = self.regions['frontal'].qubits
    cereb_q = self.regions['cerebellum'].qubits

    for i in range(min(8, len(frontal_q), len(cereb_q))):
      qc.ccx(temp_q[min(2, len(temp_q) - 1)], frontal_q[i], cereb_q[i])

    theta_slice = theta[p_idx: p_idx + len(cereb_q)]
    tau_slice = tau[p_idx: p_idx + len(cereb_q)]
    self._add_temporally_controlled_layer(qc, cereb_q, temp_q, theta_slice, tau_slice)
    p_idx += len(cereb_q)

    # Measurement
    qc.measure(qr, cr)

    # Track parameters
    self.circuit_template = qc
    self.all_parameters = list(qc.parameters)
    self.num_params = len(self.all_parameters)

    return qc

  def compute_fitness(self, counts, **task_kwargs):
    """
    Computes fitness based on task type
    """
    task_map = {
      'entropy': self.fitness_evaluator.entropy_fitness,
      'decision': self.fitness_evaluator.decision_making_fitness,
      'pattern': self.fitness_evaluator.pattern_matching_fitness,
      'memory': self.fitness_evaluator.memory_retrieval_fitness,
      'classification': self.fitness_evaluator.classification_fitness,
      'spatial': self.fitness_evaluator.spatial_reasoning_fitness,
      'working_memory': self.fitness_evaluator.working_memory_fitness,
      'attention': self.fitness_evaluator.attention_focus_fitness,
      'sequence': self.fitness_evaluator.sequence_learning_fitness,
      'problem_solving': self.fitness_evaluator.problem_solving_fitness,
      'flexibility': self.fitness_evaluator.cognitive_flexibility_fitness,
      'error_detection': self.fitness_evaluator.error_detection_fitness,
      'inhibition': self.fitness_evaluator.inhibition_control_fitness,
      'analogy': self.fitness_evaluator.analogical_reasoning_fitness,
      'reward_learning': self.fitness_evaluator.reward_learning_fitness,
      'predictive': self.fitness_evaluator.predictive_coding_fitness
    }

    fitness_func = task_map.get(self.task_type, self.fitness_evaluator.entropy_fitness)
    return fitness_func(counts, **task_kwargs)

  def _create_chaotic_mutant(self, elite_params):
    """Creates mutant using chaotic perturbation"""
    perturbation = np.zeros_like(elite_params)
    chaos_val = self.chaos_seed

    for i in range(len(elite_params)):
      chaos_val = self.chaos_r * chaos_val * (1 - chaos_val)
      perturbation[i] = (2 * chaos_val - 1) * self.mutation_strength

    self.chaos_seed = chaos_val
    mutant = elite_params + perturbation
    return np.clip(mutant, 0, 2 * np.pi)

  def _compute_lyapunov_exponent(self, num_iterations=100):
    """Computes Lyapunov exponent for chaos measurement"""
    x = self.chaos_seed
    lyap_sum = 0.0

    for _ in range(num_iterations):
      derivative = abs(self.chaos_r * (1 - 2 * x))
      if derivative > 0:
        lyap_sum += np.log(derivative)
      x = self.chaos_r * x * (1 - x)

    return lyap_sum / num_iterations

  def _adapt_mutation_strength(self, current_fitness):
    """Adapts mutation based on fitness progress"""
    improvement = current_fitness - self.best_fitness_ever

    if improvement > 1e-4:
      self.plateau_counter = 0
      self.best_fitness_ever = current_fitness
      self.mutation_strength *= 0.95
      self.mutation_strength = max(self.mutation_strength, self.min_mutation_strength)
      return "EXPLOITING"
    else:
      self.plateau_counter += 1
      if self.plateau_counter >= self.plateau_threshold:
        self.mutation_strength *= 1.2
        self.mutation_strength = min(self.mutation_strength, self.max_mutation_strength)
        return "EXPLORING"
      return "STABLE"

  def _switch_attractor(self):
    """Switches to different chaotic attractor"""
    self.current_attractor_idx = (self.current_attractor_idx + 1) % len(self.chaos_attractors)
    self.chaos_r = self.chaos_attractors[self.current_attractor_idx]
    self.chaos_seed = np.random.uniform(0.1, 0.9)
    return self.chaos_r

  def run_training(self, backend_name='ibm_brisbane', num_iterations=10, 
          population_size=8, shots=512, task_kwargs=None,
          use_dynamic_decoupling=True, resilience_level=2):
    """
    Fast evolutionary training with task-specific fitness

    Args:
      backend_name: IBM Quantum backend to use
      num_iterations: Number of training iterations
      population_size: Number of circuits to evaluate per iteration
      shots: Measurement shots per circuit
      task_kwargs: Task-specific parameters for fitness function
      use_dynamic_decoupling: Enable dynamic decoupling (DD) for error suppression
      resilience_level: Error mitigation level (0=none, 1=basic, 2=ZNE)
              Level 2 uses Zero-Noise Extrapolation for better accuracy
    """
    if task_kwargs is None:
      task_kwargs = {}

    # Build circuit if not already built
    if self.circuit_template is None:
      self.build_cognitive_circuit()

    # Initialize service
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.backend(backend_name)

    print(f"🧠 Quantum Brain Training - Task: {self.task_type.upper()}")
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")
    print(f"Circuit: {self.circuit_template.num_qubits} qubits, depth {self.circuit_template.depth()}")
    print(f"Parameters: {self.num_params} (exact count from circuit)")
    print(f"Population: {population_size} | Shots: {shots}")
    print(f"\n🛡️ Error Mitigation:")
    print(f" Resilience Level: {resilience_level}")
    if resilience_level == 0:
      print(f"  → No error mitigation")
    elif resilience_level == 1:
      print(f"  → Basic measurement error mitigation")
    elif resilience_level >= 2:
      print(f"  → Zero-Noise Extrapolation (ZNE)")
      print(f"  → Projects noise-free results from multiple noise levels")
    print(f" Dynamic Decoupling: {'ENABLED' if use_dynamic_decoupling else 'DISABLED'}")
    if use_dynamic_decoupling:
      print(f"  → Protects idle qubits from decoherence")
      print(f"  → Inserts gate sequences during idle periods")
    print("=" * 70)

    # Parameter allocation summary
    print("\n📊 Parameter Distribution:")
    for name, region in self.regions.items():
      print(f" {name:15s}: {region.param_count():4d} params ({region.function})")
    print(f" {'TOTAL':15s}: {self.num_params:4d} params")
    print("=" * 70)

    # Initialize population
    population = [np.random.uniform(0, 2 * np.pi, self.num_params) 
           for _ in range(population_size)]
    fitness_history = []

    # Configure options with enhanced error mitigation
    options = Options()
    options.execution.shots = shots
    options.optimization_level = 3
    options.resilience_level = resilience_level  # 2 = ZNE

    # Enable dynamic decoupling for real hardware
    if use_dynamic_decoupling and not backend.configuration().simulator:
      options.dynamical_decoupling.enable = True
      options.dynamical_decoupling.sequence_type = 'XY4'  # XY4 sequence (robust)
      print(f"\n⚡ Dynamic Decoupling configured: XY4 sequence")

    with Session(service=service, backend=backend) as session:
      sampler = Sampler(session=session, options=options)

      for iteration in range(num_iterations):
        iter_start = time.time()

        # Prepare circuits
        circuits = []
        for params in population:
          param_dict = {self.all_parameters[i]: params[i] 
                for i in range(len(params))}
          bound = self.circuit_template.bind_parameters(param_dict)
          circuits.append(bound)

        print(f"\n[Iteration {iteration + 1}/{num_iterations}]")
        job = sampler.run(circuits)
        result = job.result()

        # Compute fitness
        fitness_scores = []
        for i in range(population_size):
          quasi_dist = result.quasi_dists[i]
          counts = {format(k, f'0{self.total_qubits}b'): int(v * shots) 
               for k, v in quasi_dist.items()}
          fitness = self.compute_fitness(counts, **task_kwargs)
          fitness_scores.append(fitness)

        # Track best
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        fitness_history.append(best_fitness)

        # Chaos analysis
        lyapunov = self._compute_lyapunov_exponent()
        self.lyapunov_history.append(lyapunov)
        self.chaos_history.append(self.chaos_seed)

        adaptation_mode = self._adapt_mutation_strength(best_fitness)

        attractor_switched = False
        if self.plateau_counter >= self.plateau_threshold * 2:
          new_r = self._switch_attractor()
          attractor_switched = True
          self.plateau_counter = 0

        iter_time = time.time() - iter_start

        print(f"Best: {best_fitness:.4f} | Avg: {np.mean(fitness_scores):.4f} | Std: {np.std(fitness_scores):.4f}")
        print(f"Mode: {adaptation_mode:11s} | σ: {self.mutation_strength:.4f} | λ: {lyapunov:.4f}")
        if attractor_switched:
          print(f"🔄 ATTRACTOR SWITCH → r={self.chaos_r:.2f}")
        print(f"Time: {iter_time:.2f}s")

        # Evolution
        elite = population[best_idx].copy()
        new_population = [elite]

        for i in range(population_size - 1):
          if i % 4 == 0 and len(self.chaos_attractors) > 1:
            old_r = self.chaos_r
            self.chaos_r = self.chaos_attractors[(self.current_attractor_idx + i//4) % len(self.chaos_attractors)]
            mutant = self._create_chaotic_mutant(elite)
            self.chaos_r = old_r
          else:
            mutant = self._create_chaotic_mutant(elite)
          new_population.append(mutant)

        population = new_population

    print("\n" + "=" * 70)
    print(f"✓ Training Complete!")
    print(f"Final fitness: {fitness_history[-1]:.4f}")
    print(f"Improvement: {((fitness_history[-1] - fitness_history[0]) / (fitness_history[0] + 1e-10) * 100):.1f}%")
    print(f"Avg Lyapunov: {np.mean(self.lyapunov_history):.4f}")

    return population[0], fitness_history

  def quick_test(self, depth_override=None):
    """Local simulation test"""
    from qiskit_aer import AerSimulator

    if self.circuit_template is None:
      self.build_cognitive_circuit()

    print(f"🧪 Local Simulation Test - Task: {self.task_type}")
    print(f"Parameters: {self.num_params}")

    params = np.random.uniform(0, 2 * np.pi, self.num_params)
    param_dict = {self.all_parameters[i]: params[i] for i in range(len(params))}
    bound = self.circuit_template.bind_parameters(param_dict)

    simulator = AerSimulator()
    transpiled = transpile(bound, simulator)

    print(f"Depth: {transpiled.depth()} | Gates: {transpiled.size()}")

    job = simulator.run(transpiled, shots=1024)
    counts = job.result().get_counts()

    fitness = self.compute_fitness(counts)
    print(f"Fitness: {fitness:.4f}")

    top_5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 states:")
    for state, count in top_5:
      print(f" {state[:15]}...{state[-15:]}: {count:4d} ({count/1024*100:.1f}%)")

    return counts, fitness


# Example usage
if __name__ == "__main__":

  print("🧠 QUANTUM COGNITIVE SYSTEM - TASK DEMONSTRATIONS")
  print("=" * 70)

  # Example 1: Working Memory
  print("\n1️⃣ WORKING MEMORY (Miller's 7±2)")
  print("-" * 70)
  qbrain1 = FastQuantumBrain(total_qubits=133, task_type='working_memory')
  counts1, fitness1 = qbrain1.quick_test()
  print(f"→ Maintains {7} concurrent items with fitness: {fitness1:.4f}")

  # Example 2: Attention Focus
  print("\n2️⃣ ATTENTION FOCUS")
  print("-" * 70)
  qbrain2 = FastQuantumBrain(total_qubits=133, task_type='attention')
  counts2, fitness2 = qbrain2.quick_test()
  print(f"→ Selective attention fitness: {fitness2:.4f}")

  # Example 3: Cognitive Flexibility
  print("\n3️⃣ COGNITIVE FLEXIBILITY (Task Switching)")
  print("-" * 70)
  qbrain3 = FastQuantumBrain(total_qubits=133, task_type='flexibility')
  counts3, fitness3 = qbrain3.quick_test()
  print(f"→ Multi-modal adaptability: {fitness3:.4f}")

  # Example 4: Inhibition Control
  print("\n4️⃣ INHIBITION CONTROL")
  print("-" * 70)
  qbrain4 = FastQuantumBrain(total_qubits=133, task_type='inhibition')
  # Requires target and distractors - would be set in task_kwargs during training
  print("→ Resists distractors and selects target response")

  print("\n" + "=" * 70)
  print("📚 AVAILABLE COGNITIVE TASKS:")
  print("=" * 70)

  tasks_info = [
    ("entropy", "Quantum complexity (baseline)", "None"),
    ("decision", "Clear choice selection", "num_choices=4"),
    ("pattern", "Exact pattern matching", "target_pattern='...'"),
    ("memory", "Retrieve stored patterns", "memory_patterns=[...]"),
    ("classification", "Categorize into classes", "class_patterns={...}"),
    ("spatial", "Symmetry & locality", "None"),
    ("working_memory", "Maintain multiple items", "capacity=7"),
    ("attention", "Focus on region", "focus_region_size=8"),
    ("sequence", "Learn temporal order", "target_sequence=[...]"),
    ("problem_solving", "Navigate state space", "initial_state, goal_state"),
    ("flexibility", "Task switching", "num_contexts=4"),
    ("error_detection", "Avoid errors", "correct_patterns, error_patterns"),
    ("inhibition", "Resist distractors", "target_pattern, distractor_patterns"),
    ("analogy", "A:B :: C:?", "source_pattern, target_pattern"),
    ("reward_learning", "Maximize rewards", "reward_states, penalty_states"),
    ("predictive", "Prediction error", "expected_pattern")
  ]

  for i, (task, desc, kwargs) in enumerate(tasks_info, 1):
    print(f"{i:2d}. {task:18s} - {desc:30s} | kwargs: {kwargs}")

  print("\n" + "=" * 70)
  print("🚀 USAGE EXAMPLES:")
  print("=" * 70)

  print("\n# Working Memory Task:")
  print("qbrain = FastQuantumBrain(total_qubits=133, task_type='working_memory')")
  print("best, hist = qbrain.run_training(backend_name='ibm_brisbane',")
  print("                 task_kwargs={'capacity': 7})")

  print("\n# Inhibition Control Task:")
  print("qbrain = FastQuantumBrain(total_qubits=133, task_type='inhibition')")
  print("target = '1010101010...' # Target response")
  print("distractors = ['0101010101...', '1100110011...'] # Must resist")
  print("best, hist = qbrain.run_training(backend_name='ibm_brisbane',")
  print("  task_kwargs={'target_pattern': target, 'distractor_patterns': distractors})")

  print("\n# Problem Solving Task:")
  print("qbrain = FastQuantumBrain(total_qubits=133, task_type='problem_solving')")
  print("initial = '000...000' # Starting state")
  print("goal = '111...111'   # Goal state")
  print("forbidden = ['010...', '100...'] # Obstacles")
  print("best, hist = qbrain.run_training(backend_name='ibm_brisbane',")
  print("  task_kwargs={'initial_state': initial, 'goal_state': goal,")
  print("         'forbidden_states': forbidden})")

  print("\n# Reward Learning Task:")
  print("qbrain = FastQuantumBrain(total_qubits=133, task_type='reward_learning')")
  print("rewards = ['11110000...', '00001111...'] # High reward states")
  print("penalties = ['10101010...', '01010101...'] # Penalty states")
  print("best, hist = qbrain.run_training(backend_name='ibm_brisbane',")
  print("  task_kwargs={'reward_states': rewards, 'penalty_states': penalties})")

  print("\n# Predictive Coding Task:")
  print("qbrain = FastQuantumBrain(total_qubits=133, task_type='predictive')")
  print("expected = '10110110...' # Expected pattern")
  print("best, hist = qbrain.run_training(backend_name='ibm_brisbane',")
  print("  task_kwargs={'expected_pattern': expected,")
  print("         'prediction_error_threshold': 0.3})")

  print("\n" + "=" * 70)
  print("💡 COGNITIVE NEUROSCIENCE APPLICATIONS:")
  print("=" * 70)

  applications = [
    ("Working Memory", "Model prefrontal cortex capacity limits", "Neuroscience research"),
    ("Attention", "Study selective attention mechanisms", "ADHD research"),
    ("Inhibition", "Model impulse control disorders", "Clinical psychology"),
    ("Cognitive Flexibility", "Autism spectrum task-switching", "Developmental neuroscience"),
    ("Reward Learning", "Dopaminergic learning circuits", "Addiction research"),
    ("Error Detection", "Anterior cingulate monitoring", "Error-related negativity"),
    ("Predictive Coding", "Hierarchical brain models", "Perception research"),
    ("Problem Solving", "Hippocampal navigation", "Spatial cognition"),
    ("Sequence Learning", "Motor cortex temporal patterns", "Skill acquisition"),
    ("Analogical Reasoning", "Relational thinking", "Intelligence research")
  ]

  for task, mechanism, application in applications:
    print(f" • {task:22s}: {mechanism:35s} → {application}")

  print("\n" + "=" * 70)
  print("🔬 RESEARCH WORKFLOW:")
  print("=" * 70)
  print("""
1. DEFINE COGNITIVE TASK
  qbrain = FastQuantumBrain(total_qubits=133, task_type='attention')

2. TEST LOCALLY (validate before hardware)
  counts, fitness = qbrain.quick_test()

3. TRAIN ON IBM QUANTUM (optimize parameters)
  best_params, history = qbrain.run_training(
    backend_name='ibm_brisbane',
    num_iterations=20,
    population_size=8,
    shots=512,
    task_kwargs={...} # Task-specific parameters
  )

4. ANALYZE RESULTS (chaos dynamics, fitness evolution)
  qbrain.plot_chaos_analysis(history)

5. EXTRACT INSIGHTS (examine parameter distributions)
  # Which brain regions are most active?
  for name, region in qbrain.regions.items():
    print(f"{name}: {region.param_count()} parameters")
""")

  print("=" * 70)
  print("🎯 ADVANCED FEATURES:")
  print("=" * 70)
  print("""
✓ Robust parameter management (no manual indexing)
✓ Exact parameter counting (from built circuit)
✓ 16 cognitive fitness functions (task-specific)
✓ Chaotic mutation (logistic map)
✓ Adaptive exploration/exploitation
✓ Multi-attractor chaos (4 regimes)
✓ Lyapunov exponent tracking
✓ Automatic attractor switching
✓ Structured brain regions (named parameters)
✓ 133-qubit full utilization
✓ Fast batch evaluation
✓ Real-time monitoring
""")

  print("=" * 70)
  print("📊 CUSTOM TASK CREATION:")
  print("=" * 70)
  print("""
You can add your own fitness function:

class CognitiveFitnessEvaluator:
  @staticmethod
  def my_custom_task_fitness(counts, **kwargs):
    total = sum(counts.values())

    # Your custom logic here
    # Example: reward states with specific properties
    score = 0
    for state, count in counts.items():
      prob = count / total
      # Analyze state and add to score
      if some_condition(state):
        score += prob

    return score

# Use it:
qbrain.task_type = 'custom'
# Add to task_map in compute_fitness()
""")

  print("=" * 70)
  print("✅ SYSTEM READY FOR IBM QUANTUM EXECUTION")
  print("=" * 70)