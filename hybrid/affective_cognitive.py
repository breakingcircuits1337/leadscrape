"""
Affective-Cognitive state modeling and mapping utilities.

Provides:
- Emotion and Drive state containers
- Mapping from affective state to neuromodulation parameter angles
"""

from typing import Dict, List
import numpy as np


def emotion_drive_to_neuromod_angles(
    neuromod_config,
    emotion_state: Dict[str, float],
    drive_state: Dict[str, float],
    max_angles: Dict[str, float] = None,
) -> np.ndarray:
    """
    Map affective state (emotion + drives) to a neuromodulation parameter vector.

    Order must match StateDependentQuantumNeuromodulation.add_parameterized_modulation:
      1) Fear suppression: amygdala_controls -> frontal_targets[:3]  (CRY, negative angle)
      2) Interoceptive enhancement: insula_controls -> sensory_targets[:3] (CRZ, positive angle)
      3) Executive dampening: frontal_controls[:2] -> emotional_targets (CP, negative angle)

    Angles are scaled by emotion intensities:
      - fear_intensity controls fear suppression magnitude
      - interoception_intensity controls sensory enhancement
      - executive_intensity controls dampening of emotional reactivity

    Drives can optionally modulate the intensities (e.g., threat_vigilance boosts fear,
    reward_seeking reduces executive dampening).
    """
    if max_angles is None:
        max_angles = {
            "fear": np.pi / 2,        # up to 90 degrees suppression
            "interoception": np.pi / 2,  # up to 90 degrees enhancement
            "executive": np.pi / 2,   # up to 90 degrees dampening
        }

    # Extract intensities with defaults
    fear_i = float(emotion_state.get("fear", 0.0))
    intero_i = float(emotion_state.get("interoception", emotion_state.get("joy", 0.0)))
    exec_i = float(emotion_state.get("executive_control", 0.0))

    # Drives
    threat_vig = float(drive_state.get("threat_vigilance", 0.0))
    reward_seek = float(drive_state.get("reward_seeking", 0.0))
    hunger = float(drive_state.get("hunger", 0.0))

    # Simple drive-modulated intensities
    fear_i = np.clip(fear_i + 0.5 * threat_vig, 0.0, 1.0)
    intero_i = np.clip(intero_i + 0.5 * hunger, 0.0, 1.0)
    exec_i = np.clip(exec_i - 0.3 * reward_seek, 0.0, 1.0)

    params: List[float] = []

    # 1) Fear suppression (negative angle via CRY)
    for control_idx in neuromod_config.amygdala_controls:
        for target_idx in neuromod_config.frontal_targets[:3]:
            params.append(-fear_i * max_angles["fear"])

    # 2) Interoceptive enhancement (positive angle via CRZ/CRY/CP)
    for control_idx in neuromod_config.insula_controls:
        for target_idx in neuromod_config.sensory_targets[:3]:
            params.append(intero_i * max_angles["interoception"])

    # 3) Executive dampening (negative angle via CP)
    for control_idx in neuromod_config.frontal_controls[:2]:
        for target_idx in neuromod_config.emotional_targets:
            params.append(-exec_i * max_angles["executive"])

    return np.array(params, dtype=np.float32)