"""State preparation ``A`` for the discrete eta-mixture model (docs/qae_design.md §2).

    A |0>_n |0>_anc = sum_i sqrt(w_i) |i>_n ⊗ [ sqrt(1 - F_i) |0> + sqrt(F_i) |1> ]

with ``F_i = F(t | beta, eta_i)`` taken from
``DiscreteEtaModel.component_failure_probabilities``. Then

    P(anc = 1) = sum_i w_i F_i = DiscreteEtaModel.failure_probability(t).

Register layout: qubits ``0 .. n-1`` hold the eta index (Qiskit little-endian),
and the last qubit (index ``n``) is the failure ancilla.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, UCRYGate
from qiskit.quantum_info import Statevector

from qsr.models.uncertainty import DiscreteEtaModel
from qsr.validation import check_nonnegative_finite, check_positive_int

INDEX_REGISTER = "eta_index"
ANCILLA_REGISTER = "failure"


def _check_model(model: DiscreteEtaModel) -> DiscreteEtaModel:
    if not isinstance(model, DiscreteEtaModel):
        raise TypeError("model must be a DiscreteEtaModel")
    return model


def _check_mission_time(mission_time: float) -> float:
    arr = check_nonnegative_finite("mission_time", mission_time)
    if arr.ndim != 0:
        raise ValueError("mission_time must be a scalar")
    return float(arr)


def n_index_qubits(n_components: int) -> int:
    """Qubits needed to index ``n_components`` values: ``max(1, ceil(log2 K))``."""
    k = check_positive_int("n_components", n_components)
    return max(1, (k - 1).bit_length())


def padded_weights(model: DiscreteEtaModel) -> NDArray[np.float64]:
    """Model weights zero-padded to length ``2**n``; padded indices carry no probability."""
    model = _check_model(model)
    out = np.zeros(2 ** n_index_qubits(model.n_components))
    out[: model.n_components] = model.weights
    return out


def rotation_angles(model: DiscreteEtaModel, mission_time: float) -> NDArray[np.float64]:
    """RY angles ``theta_i = 2 arcsin(sqrt(F_i))``, zero-padded to length ``2**n``.

    ``RY(theta_i)|0> = sqrt(1 - F_i)|0> + sqrt(F_i)|1>``. Padded entries have
    zero weight, so their angle (0) has no effect on any probability.
    """
    model = _check_model(model)
    f = model.component_failure_probabilities(_check_mission_time(mission_time))
    angles = np.zeros(2 ** n_index_qubits(model.n_components))
    angles[: model.n_components] = 2.0 * np.arcsin(np.sqrt(np.clip(f, 0.0, 1.0)))
    return angles


def build_state_preparation(model: DiscreteEtaModel, mission_time: float) -> QuantumCircuit:
    """Circuit ``A`` (no measurements). The failure ancilla is the last qubit."""
    model = _check_model(model)
    t = _check_mission_time(mission_time)
    n = n_index_qubits(model.n_components)
    index = QuantumRegister(n, INDEX_REGISTER)
    ancilla = QuantumRegister(1, ANCILLA_REGISTER)
    qc = QuantumCircuit(index, ancilla, name="A")
    # 1. Uncertainty loading: sum_i sqrt(w_i)|i>.
    qc.append(StatePreparation(np.sqrt(padded_weights(model)), label="load w"), index)
    # 2. Failure oracle: RY(theta_i) on the ancilla, uniformly controlled by the index.
    #    UCRYGate takes the target qubit first, then the controls (little-endian index).
    qc.append(UCRYGate(rotation_angles(model, t).tolist()), [ancilla[0], *index])
    return qc


def exact_ancilla_probability(state_preparation: QuantumCircuit) -> float:
    """Noiseless statevector probability that the last qubit (failure ancilla) is ``|1>``."""
    if not isinstance(state_preparation, QuantumCircuit):
        raise TypeError("state_preparation must be a QuantumCircuit")
    if state_preparation.num_clbits:
        raise ValueError("state_preparation must not contain measurements")
    probs = Statevector(state_preparation).probabilities([state_preparation.num_qubits - 1])
    return float(probs[1])
