"""
Hybrid Quantum Bridge — Re-uploading VQC
Encodes all 22 features into 4 data qubits across 6 encoding rounds.
Total qubits: 7 (4 data + 3 readout). Total trainable params: 14.
"""

import os
import math
import pickle
import numpy as np

QSHARP_PROJECT = os.path.dirname(os.path.abspath(__file__))
QUANTUM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "quantum_params.pkl")


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class QuantumVQCBridge:
    """
    Re-uploading Variational Quantum Classifier.

    Architecture
    ------------
    n_qubits = 4
    n_readout     = 3  (one per class: DOWN / FLAT / UP)
    n_total       = 7
    n_rounds      = ceil(n_features / n_qubits) = 6

    Each round: Ry-encode chunk of features + CNOT ring
    Final layer: trainable Ry/Rz on all 7 qubits + CNOT ring
    """

    N_CLASSES    = 3
    N_READOUT    = 3

    def __init__(
        self,
        n_features: int = 22,
        n_qubits: int = 4,
        n_shots: int = 32,
    ):
        self.n_features    = n_features
        self.n_qubits      = n_qubits
        self.n_total       = n_qubits + self.N_READOUT
        self.n_rounds      = math.ceil(n_features / n_qubits)
        self.n_shots       = n_shots
        self.total_params  = self.n_rounds * self.n_total * 2

        rng = np.random.default_rng(42)
        self.params = rng.uniform(-0.1, 0.1, size=self.total_params)

        self.extra_cnots: list[list[tuple[int, int]]] = [[] for _ in range(self.n_rounds)]

        self._qsharp_available = self._try_import_qsharp()

    # Correlation-aware entanglement

    def set_entanglement_from_correlations(
        self,
        corr_matrix: np.ndarray,
        threshold: float = 0.7,
    ) -> None:
        """
        For each encoding round, find feature pairs whose |correlation| > threshold
        that are NOT already connected by the ring CNOT (non-adjacent qubits).
        Those pairs get extra CNOT gates in _run_numpy_simulator.

        corr_matrix : (n_features, n_features) Pearson correlation matrix
        threshold   : minimum |corr| to warrant a dedicated CNOT
        """
        self.extra_cnots = []
        for r in range(self.n_rounds):
            start = r * self.n_qubits
            end   = min(start + self.n_qubits, self.n_features)
            n_in  = end - start
            pairs = []
            for i in range(n_in):
                for j in range(i + 2, n_in):          # skip adjacent (i+1)
                    if i == 0 and j == n_in - 1:       # ring close already covered
                        continue
                    if abs(corr_matrix[start + i, start + j]) > threshold:
                        pairs.append((i, j))
            self.extra_cnots.append(pairs)

        total_extra = sum(len(p) for p in self.extra_cnots)
        print(f"[QuantumBridge] Correlation entanglement (|corr|>{threshold}):")
        for r, pairs in enumerate(self.extra_cnots):
            if pairs:
                pair_str = ", ".join(f"CNOT(q{c},q{t})" for c, t in pairs)
                print(f"  Round {r+1}: {pair_str}")


    def _try_import_qsharp(self) -> bool:
        try:
            import qsharp
            qsharp.init(project_root=QSHARP_PROJECT)
            self._qsharp = qsharp
            print("[QuantumBridge] Q# loaded successfully.")
            return True
        except ImportError:
            print("[QuantumBridge] WARNING: 'qsharp' not found. Using numpy simulator.")
            return False
        except Exception as exc:
            print(f"[QuantumBridge] WARNING: Q# init failed ({exc}). Using numpy simulator.")
            return False


    def _scale_features(self, x: np.ndarray) -> np.ndarray:
        """Clip and map features to [-pi, pi] for angle encoding."""
        return np.clip(x, -5.0, 5.0) * (math.pi / 5.0)
    

    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = x[0]
        if len(x) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x)}")
        return self._scale_features(x.astype(float))
    

    def _run_qsharp(self, features: np.ndarray) -> np.ndarray:
        features_list = [float(v) for v in features]
        params_list   = [float(v) for v in self.params]
        result = self._qsharp.eval(
            f"BtcQuantum.ClassExpectations("
            f"{features_list}, "
            f"{params_list}, "
            f"{self.n_qubits}, "
            f"{self.n_rounds}, "
            f"{self.n_shots})"
        )
        return np.array(result, dtype=float)
    

    def _run_numpy_simulator(self, features: np.ndarray) -> np.ndarray:
        n = self.n_total   # 7
        dim = 2 ** n       # 128 — fast!

        def apply_gate(state, gate, qubit):
            sv = state.reshape([2] * n)
            sv = np.tensordot(gate, sv, axes=([1], [qubit]))
            return np.moveaxis(sv, 0, qubit).reshape(dim)

        def apply_cnot(state, ctrl, tgt):
            sv = state.reshape([2] * n)
            idx = [slice(None)] * n
            idx[ctrl] = 1
            sub = sv[tuple(idx)]
            local = tgt if tgt < ctrl else tgt - 1
            sv[tuple(idx)] = np.flip(sub, axis=local)
            return sv.reshape(dim)

        def ry(t): c,s=math.cos(t/2),math.sin(t/2); return np.array([[c,-s],[s,c]])
        def rz(t): e=complex(math.cos(t/2),-math.sin(t/2)); return np.array([[e,0],[0,e.conjugate()]])

        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

        # Interleaved: encoding round → variational layer (repeated)
        params_per_round = self.n_total * 2
        for r in range(self.n_rounds):
            # Encode features into data qubits
            for q in range(self.n_qubits):
                idx = r * self.n_qubits + q
                if idx < self.n_features:
                    state = apply_gate(state, ry(features[idx]), q)
            # Ring CNOT — adjacent pairs
            for q in range(self.n_qubits - 1):
                state = apply_cnot(state, q, q + 1)
            state = apply_cnot(state, self.n_qubits - 1, 0)
            # Extra CNOTs — non-adjacent correlated feature pairs
            for ctrl, tgt in self.extra_cnots[r]:
                state = apply_cnot(state, ctrl, tgt)

            # Variational layer after each encoding round
            offset = r * params_per_round
            for q in range(n):
                state = apply_gate(state, ry(self.params[offset + 2 * q]),     q)
                state = apply_gate(state, rz(self.params[offset + 2 * q + 1]), q)
            for q in range(n - 1):
                state = apply_cnot(state, q, q + 1)
            state = apply_cnot(state, n - 1, 0)

        # Readout: <Z> for last 3 qubits
        probs = np.abs(state) ** 2
        sv = probs.reshape([2] * n)
        expectations = np.zeros(3)
        for cls in range(3):
            qubit = self.n_qubits + cls
            idx1 = [slice(None)] * n
            idx1[qubit] = 1
            p1 = sv[tuple(idx1)].sum()
            expectations[cls] = 1.0 - 2.0 * p1
        return expectations

    def forward(self, features: np.ndarray, use_qsharp: bool = False) -> np.ndarray:
        x = self._prepare_input(features)
        if use_qsharp and self._qsharp_available:
            return self._run_qsharp(x)
        return self._run_numpy_simulator(x)  # default: exact statevector

    def predict_proba(self, features: np.ndarray, temperature: float = 1.0, use_qsharp: bool = False) -> np.ndarray:
        return _softmax(self.forward(features, use_qsharp=use_qsharp) / temperature)

    def predict(self, features: np.ndarray, use_qsharp: bool = False) -> int:
        return int(np.argmax(self.predict_proba(features, use_qsharp=use_qsharp)))

    def spsa_gradient(self, features: np.ndarray, targets: np.ndarray,
                      epsilon: float = 0.1, n_estimates: int = 3) -> np.ndarray:
    
        class_weights = np.array([1.8, 0.7, 1.8])

        grad = np.zeros_like(self.params)
        saved = self.params.copy()

        for _ in range(n_estimates):
            delta = np.random.choice([-1.0, 1.0], size=len(self.params))

            self.params = saved + epsilon * delta
            probs_plus = self.predict_proba(features)

            self.params = saved - epsilon * delta
            probs_minus = self.predict_proba(features)

            loss_plus  = -np.sum(class_weights * targets * np.log(probs_plus  + 1e-9))
            loss_minus = -np.sum(class_weights * targets * np.log(probs_minus + 1e-9))

            grad += (loss_plus - loss_minus) / (2 * epsilon) * delta

        self.params = saved
        return grad / n_estimates

    def save(self, path: str = QUANTUM_MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "params":        self.params,
                "n_features":    self.n_features,
                "n_qubits": self.n_qubits,
                "n_shots":       self.n_shots,
            }, f)
        print(f"[QuantumBridge] Saved to {path}")

    @classmethod
    def load(cls, path: str = QUANTUM_MODEL_PATH) -> "QuantumVQCBridge":
        with open(path, "rb") as f:
            d = pickle.load(f)
        bridge = cls(n_features=d["n_features"], n_qubits=d["n_qubits"], n_shots=d["n_shots"])
        bridge.params = d["params"]
        print(f"[QuantumBridge] Loaded from {path}")
        return bridge

if __name__ == "__main__":
    print("=" * 55)
    print("  Quantum Bridge (Re-uploading) — Smoke Test")
    print("=" * 55)

    bridge = QuantumVQCBridge(n_features=22, n_qubits=4, n_shots=32)

    print(f"\nn_qubits : {bridge.n_qubits}")
    print(f"n_total       : {bridge.n_total}")
    print(f"n_rounds      : {bridge.n_rounds}")
    print(f"total_params  : {bridge.total_params}")
    print(f"Q# available  : {bridge._qsharp_available}")

    dummy = np.random.randn(22)
    probs = bridge.predict_proba(dummy)
    pred  = bridge.predict(dummy)

    classes = ["DOWN", "FLAT", "UP"]
    print(f"\nProbabilities : {probs.round(4)}")
    print(f"Prediction    : {classes[pred]}")
    print(f"Sum of probs  : {probs.sum():.6f}")
    print("\nSmoke test passed.")
