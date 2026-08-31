# Hybrid Quantum-Classical BTC Price Direction Predictor

A machine learning system that predicts Bitcoin price direction (UP / DOWN / FLAT) over **4-hour horizons** using a hybrid architecture: a **Variational Quantum Classifier (VQC)** built in **Q#** alongside a classical **Attention LSTM** baseline.

> **Academic project** — demonstrates quantum-classical hybrid ML applied to financial time-series prediction.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    Data Pipeline                    │
│  Binance API → SQLite → 18 Technical Features       │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
  ┌───────▼────────┐       ┌────────▼────────┐
  │  Classical     │       │  Hybrid Quantum │
  │  Attention     │       │  VQC  (Q# / MS  │
  │  LSTM          │       │  Quantum SDK)   │
  │                │       │                 │
  │  F1 ≈ 0.344    │       │  F1 ≈ 0.298     │
  └───────┬────────┘       └────────┬────────┘
          │                         │
          └────────────┬────────────┘
                       │
              ┌────────▼────────┐
              │   Backtester    │
              │  + Break-even   │
              │    Analysis     │
              └─────────────────┘
```

---

## Quantum VQC — Key Design Decisions

### 1. Re-uploading Encoding
Classical data is encoded into qubit rotations across **3 rounds** (18 features ÷ 6 qubits per round). Each round re-encodes new features and applies a trainable variational layer — a proven technique for encoding more information than the qubit count alone allows.

### 2. Correlation-Aware Entanglement
Features are first analysed for pairwise Pearson correlation. Correlated features are:
- **Grouped into the same encoding round** so they share quantum state
- **Entangled with targeted CNOTs** beyond the standard ring, directly between highly-correlated qubit pairs

```
Round 1  →  rsi, bb_pctb, ema_gap_fast, macd, macd_hist, slope
             Extra CNOTs: CNOT(rsi, ema_gap_fast) corr=0.82
                          CNOT(rsi, macd)          corr=0.76
                          CNOT(ema_gap_fast, macd_hist) corr=0.79
                          CNOT(macd, slope)         corr=0.88

Round 2  →  momentum, realized_vol, bb_width, vol_regime, log_return_1, high_low_range
             Extra CNOTs: CNOT(realized_vol, vol_regime) corr=0.83

Round 3  →  log_volume, volume_zscore, hour_sin, hour_cos, dow_sin, dow_cos
             (ring only — all high-corr pairs are adjacent)
```

The ring CNOT handles adjacent pairs; extra CNOTs handle non-adjacent correlated pairs that the ring misses.

### 3. Circuit Architecture

```
9 qubits total  (6 data + 3 readout)
3 encoding rounds
54 trainable parameters  (3 rounds × 9 qubits × 2 gates)

Per round:
  Ry(feature_i) on each data qubit
  → Ring CNOT (adjacent entanglement)
  → Extra CNOTs (correlation-aware entanglement)
  → Variational layer: Ry + Rz on all 9 qubits
  → Ring CNOT
```

Readout: `⟨Z⟩` measured on the 3 readout qubits → softmax → class probabilities.

### 4. Training
- **SPSA gradient** (Simultaneous Perturbation Stochastic Approximation): only 2 forward passes per gradient step regardless of parameter count
- **5 SPSA estimates** averaged per step for lower-variance gradients
- **Manual Adam** optimizer on numpy parameters
- **Balanced class weights** [DOWN=1.83, FLAT=0.56, UP=1.53] to handle the ~60% FLAT imbalance
- **Numpy statevector simulator** for training (exact, fast); Q# reserved for inference

---

## Classical LSTM Baseline

```
Input  [batch, 96, 18]           96 candles × 18 features
  ↓
LSTM   (2 layers, 128 hidden, dropout=0.30)
  ↓
Attention  (QK scaled dot-product)
  ↓
LayerNorm + Dropout
  ↓
Dense  (256 → 3)
  ↓
Softmax  → [P(DOWN), P(FLAT), P(UP)]
```

- **Focal loss** (γ=1.0) to focus on hard examples
- **Expanding-window cross-validation** (3 folds, 48-candle embargo gap)
- **Temperature calibration** on validation set

---

## Feature Engineering

18 features selected after correlation analysis (dropped 4 pairs with corr > 0.9):

| Round | Features | Cluster |
|-------|----------|---------|
| 1 | rsi, bb_pctb, ema_gap_fast, macd, macd_hist, slope | RSI/EMA + MACD/trend |
| 2 | momentum, realized_vol, bb_width, vol_regime, log_return_1, high_low_range | Volatility + return |
| 3 | log_volume, volume_zscore, hour_sin, hour_cos, dow_sin, dow_cos | Volume + time cycles |

Dropped (corr > 0.9): `close_open` (=`log_return_1`), `macd_signal`, `atr_norm`, `ema_gap_slow`

---

## Results

| Model | Macro-F1 | Random Baseline |
|-------|----------|----------------|
| Attention LSTM | 0.344 | 0.333 |
| Quantum VQC | 0.298 | 0.333 |

**Break-even analysis** (0.1% fee per side):

| Avg 4h BTC move | Win rate needed to profit |
|-----------------|--------------------------|
| 0.3% | 83% |
| 0.5% | 70% |
| 1.0% | 60% |
| 2.0% | 55% |

The models' win rates (24–28%) reflect the difficulty of financial prediction, not a bug. In flat/choppy markets, no directional model consistently overcomes transaction costs — this is the honest result.

---

## Project Structure

```
.
├── main.py                  # CLI entry point
├── config.py                # All hyperparameters
├── data_collector.py        # Binance API + SQLite
├── feature_engineer.py      # 18 technical features + adaptive labeling
├── model.py                 # Attention LSTM + Focal Loss
├── trainer.py               # LSTM trainer + QuantumTrainer (SPSA, Adam)
├── quantum_bridge.py        # Python↔Q# bridge + numpy simulator
├── backtest.py              # Trading simulation + break-even analysis
├── live_predictor.py        # Real-time inference
├── src/
│   └── QuantumClassifier.qs # Q# circuit (correlation-aware CNOTs)
├── BtcQuantum.csproj        # .NET project file for Q#
├── requirements.txt
└── models/                  # Saved after training (git-ignored)
    ├── lstm_model.pt
    ├── quantum_params.pkl
    ├── scaler.pkl
    └── calibration.pkl
```

---

## Setup

### Prerequisites
- Python 3.9+
- [.NET SDK 8+](https://dotnet.microsoft.com/download) (for Q# — optional, falls back to numpy simulator)

### Install

```bash
git clone https://github.com/<your-username>/btc-quantum-predictor.git
cd btc-quantum-predictor
pip install -r requirements.txt
```

### Run

```bash
# 1. Collect data (30 days of 5-min BTC/USDT candles from Binance)
python main.py collect --days 30

# 2a. Train classical LSTM
python main.py train

# 2b. Train hybrid quantum VQC
python main.py train --quantum

# 3. Backtest both models
python backtest.py --model both

# 4. Live prediction (requires trained LSTM)
python main.py predict --single
```

---

## Q# Integration

The quantum circuit is defined in `src/QuantumClassifier.qs` using Microsoft's Quantum Development Kit.

```bash
# Check Q# availability
python -c "import qsharp; print(qsharp.__version__)"
```

If Q# is unavailable, the bridge automatically falls back to an exact **numpy statevector simulator** — identical results, no .NET required. Training always uses numpy (fast); Q# is used for inference when available.

To install Q# support:
```bash
# .NET SDK 8+ required first
pip install qsharp
dotnet build BtcQuantum.csproj
```

---

## Configuration

All settings are in `config.py`:

```python
# Quantum circuit
quantum_config.n_data_qubits = 6     # data encoding qubits
quantum_config.spsa_estimates = 5    # gradient estimates per step
quantum_config.max_epochs = 200
quantum_config.learning_rate = 0.004

# LSTM
model_config.hidden_size = 128
model_config.num_layers = 2
model_config.use_attention = True

# Cross-validation
cv_config.n_splits = 3
cv_config.embargo_min_gap = 48       # candles between train/test (= 4h)
```

---

## Disclaimer

Educational project only. Not financial advice. Cryptocurrency trading involves significant risk of loss.
