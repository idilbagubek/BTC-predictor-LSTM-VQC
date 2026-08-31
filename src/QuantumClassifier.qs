namespace BtcQuantum {
    import Std.Convert.IntAsDouble;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Measurement;

    // Encode one chunk of features into data qubits.
    // After angle encoding, applies:
    //   1. Ring CNOT (adjacent pairs) — general entanglement
    //   2. Extra CNOTs for non-adjacent correlated feature pairs (round-specific)
    //
    // Correlation-aware extra CNOTs (|corr| > 0.7, non-adjacent):
    //   Round 0: CNOT(q0,q2) rsi↔ema_gap_fast 0.82
    //            CNOT(q0,q3) rsi↔macd         0.76
    //            CNOT(q2,q4) ema_gap_fast↔macd_hist 0.79
    //            CNOT(q3,q5) macd↔slope        0.88
    //   Round 1: CNOT(q1,q3) realized_vol↔vol_regime 0.83
    //   Round 2: (none — ring covers all high-corr pairs)
    operation EncodingRound(
        features : Double[],
        qubits   : Qubit[],
        n_data   : Int,
        round    : Int
    ) : Unit is Adj + Ctl {
        let n_features = Length(features);

        // Angle encoding
        for q in 0 .. n_data - 1 {
            let idx = round * n_data + q;
            if idx < n_features {
                Ry(features[idx], qubits[q]);
            }
        }

        // Ring CNOT
        for q in 0 .. n_data - 2 {
            CNOT(qubits[q], qubits[q + 1]);
        }
        CNOT(qubits[n_data - 1], qubits[0]);

        // Extra correlation-aware CNOTs
        if round == 0 {
            CNOT(qubits[0], qubits[2]);   // rsi ↔ ema_gap_fast  (0.82)
            CNOT(qubits[0], qubits[3]);   // rsi ↔ macd          (0.76)
            CNOT(qubits[2], qubits[4]);   // ema_gap_fast ↔ macd_hist (0.79)
            CNOT(qubits[3], qubits[5]);   // macd ↔ slope        (0.88)
        }
        if round == 1 {
            CNOT(qubits[1], qubits[3]);   // realized_vol ↔ vol_regime (0.83)
        }
        // round == 2: ring sufficient
    }

    // Trainable variational layer on ALL qubits.
    // params: [Ry_0, Rz_0, Ry_1, Rz_1, ..., Ry_{n-1}, Rz_{n-1}]
    operation VariationalLayer(
        params : Double[],
        qubits : Qubit[]
    ) : Unit is Adj + Ctl {
        let n = Length(qubits);
        for q in 0 .. n - 1 {
            Ry(params[2 * q],     qubits[q]);
            Rz(params[2 * q + 1], qubits[q]);
        }
        for q in 0 .. n - 2 {
            CNOT(qubits[q], qubits[q + 1]);
        }
        CNOT(qubits[n - 1], qubits[0]);
    }

    // Interleaved circuit: each encoding round is followed by a variational layer.
    // params shape: [n_rounds × n_total × 2]  (row-major, one layer per round)
    operation ExpectationZ(
        features  : Double[],
        params    : Double[],
        n_data    : Int,
        n_rounds  : Int,
        qubit_idx : Int,
        n_shots   : Int
    ) : Double {
        let n_total = n_data + 3;
        let params_per_round = n_total * 2;
        mutable ones = 0;

        for _ in 1 .. n_shots {
            use qubits = Qubit[n_total];

            for round in 0 .. n_rounds - 1 {
                EncodingRound(features, qubits, n_data, round);
                let offset = round * params_per_round;
                let round_params = params[offset .. offset + params_per_round - 1];
                VariationalLayer(round_params, qubits);
            }

            let res = M(qubits[qubit_idx]);
            if res == One { set ones += 1; }
            ResetAll(qubits);
        }

        return 2.0 * IntAsDouble(ones) / IntAsDouble(n_shots) - 1.0;
    }

    // Returns <Z> expectations for 3 readout qubits → [P(DOWN), P(FLAT), P(UP)]
    operation ClassExpectations(
        features : Double[],
        params   : Double[],
        n_data   : Int,
        n_rounds : Int,
        n_shots  : Int
    ) : Double[] {
        let readout_start = n_data;
        mutable expectations = [0.0, 0.0, 0.0];
        for cls in 0 .. 2 {
            set expectations w/= cls <-
                ExpectationZ(features, params, n_data, n_rounds,
                             readout_start + cls, n_shots);
        }
        return expectations;
    }
}
