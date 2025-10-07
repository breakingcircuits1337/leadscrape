from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes, EfficientSU2, TwoLocal
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Options
from qiskit.primitives import BackendEstimator
import numpy as np
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import networkx as nx


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
        if total == 0: return 0
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1
        return entropy / max_entropy if max_entropy > 0 else 0

    @staticmethod
    def pattern_matching_fitness(counts, target_pattern):
        """Task: Match specific output pattern"""
        total = sum(counts.values())
        if total == 0: return 0
        target_count = counts.get(target_pattern, 0)
        return target_count / total

    @staticmethod
    def decision_making_fitness(counts, num_choices=4):
        """
        Task: Decision-making
        Rewards clear, decisive outputs (low entropy within top choices)
        """
        total = sum(counts.values())
        if total == 0: return 0
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
            max_top_entropy = np.log2(num_choices) if num_choices > 1 else 1
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
        if total == 0: return 0
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
        if total == 0: return 0

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
        if total == 0: return 0

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
            for i in range(1, len(bitstring)):
                if bitstring[i] != bitstring[i-1]:
                    runs += 1
            
            # Longer runs = more local structure
            avg_run_length = len(bitstring) / (runs + 1)
            locality_score += prob * (avg_run_length / len(bitstring))

        return 0.5 * symmetry_score + 0.5 * locality_score


class FastQuantumBrain:
    """High-performance quantum cognitive system using 133 qubits"""

    def __init__(self, total_qubits=133, task_type='entropy'):
        self.total_qubits = total_qubits
        self.task_type = task_type

        # Define brain regions with structured parameters
        self.regions = {
            'thalamus': BrainRegion('thalamus', list(range(0, 8)), 'input_routing', depth=1),
            'occipital': BrainRegion('occipital', list(range(8, 28)), 'visual_processing', depth=2),
            'parietal': BrainRegion('parietal', list(range(28, 48)), 'spatial_integration', depth=2),
            'temporal': BrainRegion('temporal', list(range(48, 73)), 'memory_processing', depth=2),
            'hippocampus': BrainRegion('hippocampus', list(range(73, 88)), 'memory_formation', depth=2),
            'amygdala': BrainRegion('amygdala', list(range(88, 96)), 'emotional_tagging', depth=1),
            'frontal': BrainRegion('frontal', list(range(96, 121)), 'decision_making', depth=3),
            'cerebellum': BrainRegion('cerebellum', list(range(121, 133)), 'motor_control', depth=1)
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

    def _add_region_layer(self, qc, region):
        """Adds a processing layer for a specific brain region"""
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

            # Entanglement layer (linear with parametric Rz)
            if len(qubits) > 1:
                for i in range(len(qubits) - 1):
                    qc.cx(qubits[i], qubits[i + 1])
                    if ent_idx < len(ent_params):
                        qc.rz(ent_params[ent_idx], qubits[i + 1])
                        ent_idx += 1

    def build_cognitive_circuit(self):
        """Builds the complete cognitive circuit with structured parameters"""
        qr = QuantumRegister(self.total_qubits, 'brain')
        cr = ClassicalRegister(self.total_qubits, 'measure')
        qc = QuantumCircuit(qr, cr)

        # PHASE 1: Input encoding (Thalamus)
        thal = self.regions['thalamus']
        for q in thal.qubits:
            qc.h(q)
        self._add_region_layer(qc, thal)

        # PHASE 2: Sensory processing (Occipital & Parietal in parallel)
        occ = self.regions['occipital']
        par = self.regions['parietal']

        for i in range(min(len(thal.qubits), len(occ.qubits), len(par.qubits))):
            qc.cx(thal.qubits[i], occ.qubits[i])
            qc.cx(thal.qubits[i], par.qubits[i])

        self._add_region_layer(qc, occ)
        self._add_region_layer(qc, par)

        # PHASE 3: Memory system
        temp = self.regions['temporal']
        hippo = self.regions['hippocampus']
        amyg = self.regions['amygdala']

        for i in range(min(10, len(occ.qubits), len(temp.qubits))):
            qc.cx(occ.qubits[i], temp.qubits[i])
        self._add_region_layer(qc, temp)

        for i in range(min(len(temp.qubits), len(hippo.qubits))):
            qc.cx(temp.qubits[i], hippo.qubits[i])
        self._add_region_layer(qc, hippo)

        for i in range(min(len(hippo.qubits), len(amyg.qubits))):
            qc.cx(hippo.qubits[i], amyg.qubits[i])
        self._add_region_layer(qc, amyg)

        # PHASE 4: Executive function (Frontal)
        front = self.regions['frontal']
        
        sources = {'hippo': hippo.qubits, 'amyg': amyg.qubits, 'par': par.qubits}
        for i in range(min(8, len(sources['hippo']), len(front.qubits))):
             qc.cx(sources['hippo'][i], front.qubits[i])
        for i in range(min(8, len(sources['amyg']), len(front.qubits)-8)):
             qc.cx(sources['amyg'][i], front.qubits[i+8])
        for i in range(min(8, len(sources['par']), len(front.qubits)-16)):
             qc.cx(sources['par'][i], front.qubits[i+16])
        self._add_region_layer(qc, front)

        # PHASE 5: Motor control (Cerebellum)
        cereb = self.regions['cerebellum']
        for i in range(min(8, len(front.qubits), len(cereb.qubits))):
            qc.cx(front.qubits[i], cereb.qubits[i])
        self._add_region_layer(qc, cereb)

        qc.measure(qr, cr)

        self.circuit_template = qc
        self.all_parameters = list(qc.parameters)
        self.num_params = len(self.all_parameters)
        return qc

    def compute_fitness(self, counts, **task_kwargs):
        """Computes fitness based on task type"""
        fitness_map = {
            'entropy': self.fitness_evaluator.entropy_fitness,
            'decision': self.fitness_evaluator.decision_making_fitness,
            'pattern': self.fitness_evaluator.pattern_matching_fitness,
            'memory': self.fitness_evaluator.memory_retrieval_fitness,
            'classification': self.fitness_evaluator.classification_fitness,
            'spatial': self.fitness_evaluator.spatial_reasoning_fitness,
        }
        fitness_func = fitness_map.get(self.task_type, self.fitness_evaluator.entropy_fitness)
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
        x, lyap_sum = self.chaos_seed, 0.0
        for _ in range(num_iterations):
            derivative = abs(self.chaos_r * (1 - 2 * x))
            if derivative > 0:
                lyap_sum += np.log(derivative)
            x = self.chaos_r * x * (1 - x)
        return lyap_sum / num_iterations

    def _adapt_mutation_strength(self, current_fitness):
        """Adapts mutation based on fitness progress"""
        if current_fitness > self.best_fitness_ever + 1e-4:
            self.plateau_counter = 0
            self.best_fitness_ever = current_fitness
            self.mutation_strength = max(self.mutation_strength * 0.95, self.min_mutation_strength)
            return "EXPLOITING"
        else:
            self.plateau_counter += 1
            if self.plateau_counter >= self.plateau_threshold:
                self.mutation_strength = min(self.mutation_strength * 1.2, self.max_mutation_strength)
                return "EXPLORING"
            return "STABLE"

    def _switch_attractor(self):
        """Switches to different chaotic attractor"""
        self.current_attractor_idx = (self.current_attractor_idx + 1) % len(self.chaos_attractors)
        self.chaos_r = self.chaos_attractors[self.current_attractor_idx]
        self.chaos_seed = np.random.uniform(0.1, 0.9)

    def run_training(self, backend_name='ibm_brisbane', num_iterations=10, population_size=8, shots=512, task_kwargs=None):
        """Fast evolutionary training with task-specific fitness"""
        if task_kwargs is None: task_kwargs = {}
        if self.circuit_template is None: self.build_cognitive_circuit()

        service = QiskitRuntimeService(channel="ibm_quantum")
        backend = service.backend(backend_name)

        print(f"🧠 Quantum Brain Training - Task: {self.task_type.upper()}")
        print(f"Backend: {backend.name} | Circuit: {self.circuit_template.num_qubits}q, Depth {self.circuit_template.depth()}, Params {self.num_params}")
        print(f"Population: {population_size} | Shots: {shots}")
        print("=" * 70)

        population = [np.random.uniform(0, 2 * np.pi, self.num_params) for _ in range(population_size)]
        fitness_history = []

        options = Options(execution={"shots": shots}, optimization_level=3, resilience_level=1)

        with Session(service=service, backend=backend) as session:
            sampler = Sampler(session=session, options=options)
            for iteration in range(num_iterations):
                iter_start = time.time()
                
                circuits = [self.circuit_template.bind_parameters(dict(zip(self.all_parameters, params))) for params in population]
                
                print(f"\n[Iteration {iteration + 1}/{num_iterations}] Submitting {len(circuits)} circuits...")
                job = sampler.run(circuits)
                result = job.result()

                fitness_scores = []
                for i in range(population_size):
                    quasi_dist = result.quasi_dists[i]
                    counts = {f'{k:0{self.total_qubits}b}': int(v * shots) for k, v in quasi_dist.items()}
                    fitness_scores.append(self.compute_fitness(counts, **task_kwargs))

                best_idx = np.argmax(fitness_scores)
                best_fitness = fitness_scores[best_idx]
                fitness_history.append(best_fitness)
                
                self.lyapunov_history.append(self._compute_lyapunov_exponent())
                self.chaos_history.append(self.chaos_seed)
                adaptation_mode = self._adapt_mutation_strength(best_fitness)

                if self.plateau_counter >= self.plateau_threshold * 2:
                    self._switch_attractor()
                    print(f"🔄 ATTRACTOR SWITCH → r={self.chaos_r:.2f}")
                    self.plateau_counter = 0

                iter_time = time.time() - iter_start
                print(f"Best: {best_fitness:.4f} | Avg: {np.mean(fitness_scores):.4f} | Mode: {adaptation_mode:11s} | σ: {self.mutation_strength:.4f} | λ: {self.lyapunov_history[-1]:.4f} | Time: {iter_time:.2f}s")
                
                elite = population[best_idx].copy()
                population = [elite] + [self._create_chaotic_mutant(elite) for _ in range(population_size - 1)]

        print("\n" + "=" * 70 + "\n✓ Training Complete!")
        print(f"Final fitness: {fitness_history[-1]:.4f} | Improvement: {((fitness_history[-1] - fitness_history[0]) / (fitness_history[0] + 1e-10) * 100):.1f}%")
        return population[0], fitness_history

    def quick_test(self):
        """Local simulation test"""
        from qiskit_aer import AerSimulator
        if self.circuit_template is None: self.build_cognitive_circuit()

        print(f"🧪 Local Simulation Test - Task: {self.task_type} | Parameters: {self.num_params}")
        params = np.random.uniform(0, 2 * np.pi, self.num_params)
        bound = self.circuit_template.bind_parameters(dict(zip(self.all_parameters, params)))
        
        simulator = AerSimulator()
        transpiled = transpile(bound, simulator)
        print(f"Depth: {transpiled.depth()} | Gates: {transpiled.size()}")
        
        counts = simulator.run(transpiled, shots=1024).result().get_counts()
        fitness = self.compute_fitness(counts)
        print(f"Fitness: {fitness:.4f}")

        top_5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 states:")
        for state, count in top_5:
            print(f"  {state[:15]}...{state[-15:]}: {count:4d} ({count/1024*100:.1f}%)")
        return counts, fitness


class QuantumVisualizer:
    """Tools for visualizing the quantum brain's structure and performance."""
    def __init__(self, brain: FastQuantumBrain):
        self.brain = brain
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_fitness_history(self, fitness_history):
        """Plots the fitness score over training iterations."""
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history, marker='o', linestyle='-', label='Best Fitness per Iteration')
        
        # Calculate and plot moving average
        window = 5
        if len(fitness_history) >= window:
            moving_avg = np.convolve(fitness_history, np.ones(window)/window, mode='valid')
            plt.plot(np.arange(window-1, len(fitness_history)), moving_avg, linestyle='--', color='red', label=f'{window}-iteration Moving Avg')

        plt.title('🏆 Fitness History', fontsize=16)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Fitness Score', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_chaos_dynamics(self, lyapunov_history, chaos_history):
        """Visualizes the chaotic parameters of the optimizer."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot Lyapunov exponent
        ax1.plot(lyapunov_history, marker='.', linestyle='-', color='purple', label='Lyapunov Exponent (λ)')
        ax1.axhline(0, color='grey', linestyle='--', linewidth=1)
        ax1.set_title('📈 Chaos Dynamics During Training', fontsize=16)
        ax1.set_ylabel('Lyapunov Exponent (λ)', fontsize=12)
        ax1.text(0.01, 0.05, 'λ > 0 → Chaotic (Exploration)', transform=ax1.transAxes, color='green')
        ax1.text(0.01, 0.15, 'λ < 0 → Stable (Exploitation)', transform=ax1.transAxes, color='blue')
        ax1.legend()

        # Plot Chaos Seed Trajectory
        ax2.plot(chaos_history, marker='', linestyle='-', color='orange', label='Chaos Seed Trajectory')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Chaos Seed Value', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_parameter_distribution(self):
        """Creates a bar chart of parameter allocation per brain region."""
        if self.brain.num_params == 0:
            self.brain.build_cognitive_circuit()

        names = list(self.brain.regions.keys())
        counts = [r.param_count() for r in self.brain.regions.values()]
        
        plt.figure(figsize=(12, 7))
        bars = plt.barh(names, counts, color=plt.cm.viridis(np.linspace(0, 1, len(names))))
        plt.xlabel('Number of Trainable Parameters', fontsize=12)
        plt.ylabel('Brain Region', fontsize=12)
        plt.title('📊 Parameter Distribution Across Brain Regions', fontsize=16)
        plt.gca().invert_yaxis() # Display thalamus at the top

        for bar in bars:
            plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                     f'{bar.get_width()}', va='center', ha='left')
        
        plt.show()

    def draw_brain_connectivity(self):
        """Draws a graph of the brain's inter-regional connections."""
        G = nx.DiGraph()
        
        # Add nodes with qubit count as an attribute for sizing
        for name, region in self.brain.regions.items():
            G.add_node(name, qubits=len(region.qubits))
            
        # Manually define edges based on the circuit build logic
        edges = [
            ('thalamus', 'occipital'), ('thalamus', 'parietal'),
            ('occipital', 'temporal'), ('parietal', 'frontal'),
            ('temporal', 'hippocampus'), ('hippocampus', 'amygdala'),
            ('hippocampus', 'frontal'), ('amygdala', 'frontal'),
            ('frontal', 'cerebellum')
        ]
        G.add_edges_from(edges)
        
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)
        
        node_sizes = [d['qubits'] * 200 for _, d in G.nodes(data=True)]
        node_colors = plt.cm.plasma(np.linspace(0, 1, G.number_of_nodes()))

        nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors,
                font_size=12, font_weight='bold', width=2.0, edge_color='grey', arrowsize=20)
        
        plt.title('🧠 Quantum Brain Connectivity Graph', fontsize=20)
        plt.show()


# Example usage
if __name__ == "__main__":

    # Example 1: Decision-making task
    print("=" * 70)
    print("EXAMPLE 1: Decision-Making Task")
    print("=" * 70)
    qbrain_decision = FastQuantumBrain(total_qubits=133, task_type='decision')
    counts, fitness = qbrain_decision.quick_test()

    # Example 2: Spatial reasoning task
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Spatial Reasoning Task")
    print("=" * 70)
    qbrain_spatial = FastQuantumBrain(total_qubits=133, task_type='spatial')
    counts, fitness = qbrain_spatial.quick_test()
    
    # --- Visualization Example ---
    print("\n" + "=" * 70)
    print("VISUALIZATION EXAMPLES")
    print("=" * 70)
    
    # Use one of the created brains for visualization
    visualizer = QuantumVisualizer(qbrain_decision)

    # 1. Parameter Distribution
    print("Displaying parameter distribution chart...")
    visualizer.plot_parameter_distribution()

    # 2. Brain Connectivity Graph
    print("Displaying brain connectivity graph...")
    visualizer.draw_brain_connectivity()
    
    # 3. Fitness & Chaos (using dummy data for demonstration)
    print("\nPlotting fitness and chaos dynamics with dummy data...")
    print("After a real training run, you would pass the actual history from run_training().")
    dummy_fitness = np.linspace(0.4, 0.85, 20) + np.random.randn(20) * 0.05
    dummy_lyapunov = np.random.uniform(0.1, 0.6, 20)
    dummy_chaos_seed = np.random.rand(20)
    
    visualizer.plot_fitness_history(dummy_fitness)
    visualizer.plot_chaos_dynamics(dummy_lyapunov, dummy_chaos_seed)


    print("\n" + "=" * 70)
    print("Ready for IBM Quantum!")
    print("=" * 70)
    print("\nTo run on hardware:")
    print("qbrain = FastQuantumBrain(total_qubits=133, task_type='decision')")
    print("best_params, history = qbrain.run_training(")
    print("    backend_name='ibm_brisbane',")
    print("    num_iterations=20,")
    print("    population_size=8,")
    print("    shots=512,")
    print("    task_kwargs={'num_choices': 4}")
    print(")")
    print("\n# Then visualize the real results:")
    print("visualizer = QuantumVisualizer(qbrain)")
    print("visualizer.plot_fitness_history(history)")
    print("visualizer.plot_chaos_dynamics(qbrain.lyapunov_history, qbrain.chaos_history)")