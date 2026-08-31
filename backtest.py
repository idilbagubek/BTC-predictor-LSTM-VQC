"""
Usage:
    python backtest.py --model lstm       # LSTM backtest
    python backtest.py --model quantum    # Quantum VQC backtest
    python backtest.py --model both       # Side-by-side comparison (default)
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from config import (
    feature_config, label_config, model_config,
    MODEL_PATH, SCALER_PATH, CALIBRATION_PATH,
)
from data_collector import load_candles_as_dataframe
from feature_engineer import FeatureEngineer, build_sequences

FEE = 0.001        
INITIAL_CAPITAL = 10_000.0
TEST_FRACTION = 0.20  

LSTM_FEATURES_22 = [
    "log_return_1", "high_low_range", "close_open", "log_volume",
    "rsi", "macd", "macd_signal", "macd_hist",
    "atr_norm", "bb_width", "bb_pctb", "ema_gap_fast", "ema_gap_slow",
    "momentum", "realized_vol", "volume_zscore",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "slope", "vol_regime",
]

# Helpers

def load_data(feat_cols=None):
    if feat_cols is None:
        feat_cols = feature_config.feature_names

    df = load_candles_as_dataframe()
    fe = FeatureEngineer()
    df_feat = fe.compute_features(df)
    labels = fe.compute_labels(df_feat, use_adaptive=label_config.use_adaptive_epsilon)

    original = feature_config.feature_names[:]
    feature_config.feature_names[:] = feat_cols
    X, y, timestamps = build_sequences(df_feat, labels, include_current_candle=True)
    feature_config.feature_names[:] = original

    valid_mask = ~df_feat[feat_cols].isnull().any(axis=1)
    df_valid = df_feat[valid_mask].reset_index(drop=True)

    n = len(y)
    split = int(n * (1 - TEST_FRACTION))
    return (
        X[split:], y[split:], timestamps[split:],
        df_valid.iloc[split + feature_config.window_size - 1:].reset_index(drop=True),
        fe,
    )


def get_future_return(df_valid: pd.DataFrame, idx: int) -> float:
   
    horizon = label_config.horizon_candles
    if idx + horizon >= len(df_valid):
        return 0.0
    entry = df_valid.iloc[idx]["close"]
    exit_ = df_valid.iloc[idx + horizon]["close"]
    return (exit_ - entry) / entry


# LSTM 

def predict_lstm(X_test: np.ndarray, fe: FeatureEngineer, feat_cols: list):
    import torch
    from model import AttentionLSTMClassifier, LSTMClassifier

    device = "cpu"
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})

    saved_input_size = checkpoint["model_state_dict"]["lstm.weight_ih_l0"].shape[1]
    orig_input = model_config.input_size
    model_config.input_size = saved_input_size

    if cfg.get("use_attention", True):
        model = AttentionLSTMClassifier(
            attention_variant=cfg.get("attention_variant", model_config.attention_variant)
        ).to(device)
    else:
        model = LSTMClassifier().to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model_config.input_size = orig_input  # restore

    temperature = 1.0
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH, "rb") as f:
            cal = pickle.load(f)
            temperature = cal.get("temperature", 1.0)

    fe.load_scaler()
    X_scaled = X_test.copy().astype(np.float32)
    for fi, col in enumerate(feat_cols):
        mean = fe.scaler_params["mean"][col]
        std  = fe.scaler_params["std"][col]
        X_scaled[:, :, fi] = (X_scaled[:, :, fi] - mean) / std

    all_probs = []
    batch = 256
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch):
            xb = torch.tensor(X_scaled[i:i+batch], dtype=torch.float32)
            logits = model(xb).cpu().numpy()
            scaled = logits / temperature
            e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            all_probs.append(e / e.sum(axis=1, keepdims=True))

    probs = np.vstack(all_probs)
    preds = np.argmax(probs, axis=1)
    return probs, preds


# VQC

def predict_quantum(X_test: np.ndarray):
    """Return (probs array, preds array) using saved VQC params."""
    from quantum_bridge import QuantumVQCBridge
    from config import quantum_config

    bridge = QuantumVQCBridge.load()

    # Normalise using quantum scaler
    q_scaler_path = os.path.join(os.path.dirname(MODEL_PATH), "quantum_scaler.pkl")
    if os.path.exists(q_scaler_path):
        with open(q_scaler_path, "rb") as f:
            sc = pickle.load(f)
        mean, std = sc["mean"], sc["std"]
    else:
        last = X_test[:, -1, :]
        mean, std = last.mean(0), last.std(0) + 1e-8

    last_steps = X_test[:, -1, :]
    last_norm = (last_steps - mean) / std

    all_probs = []
    for i in range(len(last_norm)):
        p = bridge.predict_proba(last_norm[i])
        all_probs.append(p)

    probs = np.array(all_probs)
    preds = np.argmax(probs, axis=1)
    return probs, preds


# Simulate 

def simulate(
    preds: np.ndarray,
    probs: np.ndarray,
    df_valid: pd.DataFrame,
    model_name: str,
    confidence_threshold: float = 0.42,
    long_only: bool = False,
) -> dict:
   
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trade_returns = []
    n_trades = {"UP": 0, "DOWN": 0, "FLAT": 0, "skipped_low_conf": 0}

    for i, (pred, prob) in enumerate(zip(preds, probs)):
        confidence = prob.max()

        # Skip low-confidence predictions
        if confidence < confidence_threshold:
            n_trades["skipped_low_conf"] += 1
            equity_curve.append(capital)
            continue

        raw_ret = get_future_return(df_valid, i)
        label_name = ["DOWN", "FLAT", "UP"][pred]
        n_trades[label_name] += 1

        if pred == 2:   # UP → LONG
            net = raw_ret - 2 * FEE
        elif pred == 0 and not long_only:  # DOWN → SHORT
            net = -raw_ret - 2 * FEE
        else:           
            equity_curve.append(capital)
            continue

        pnl = capital * net
        capital += pnl
        trade_returns.append(net)
        equity_curve.append(capital)

    equity = np.array(equity_curve)
    rets = np.array(trade_returns) if trade_returns else np.array([0.0])

    total_ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate  = (rets > 0).mean() * 100 if len(rets) > 1 else 0.0
    avg_ret   = rets.mean() * 100
    sharpe    = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252 * 6) if len(rets) > 1 else 0.0
    peak      = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_dd    = drawdowns.min() * 100

    return {
        "model": model_name,
        "n_trades": len(trade_returns),
        "trade_breakdown": n_trades,
        "total_return_pct": total_ret,
        "win_rate_pct": win_rate,
        "avg_trade_pct": avg_ret,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "final_capital": capital,
        "equity_curve": equity,
    }


def buy_and_hold(df_valid: pd.DataFrame) -> dict:
    """BTC buy-and-hold baseline over the same test period."""
    if len(df_valid) == 0:
        return {}
    entry = df_valid.iloc[0]["close"]
    exit_ = df_valid.iloc[-1]["close"]
    ret = (exit_ - entry) / entry * 100
    return {"model": "Buy & Hold BTC", "total_return_pct": ret}


def break_even_win_rate(avg_abs_return_pct: float, fee_pct: float = 0.2) -> float:
    """Min win rate needed to profit given avg trade size and round-trip fee."""
    r = avg_abs_return_pct / 100
    f = fee_pct / 100
    if r <= f:
        return float("nan")
    return 0.5 + f / (2 * r)


def print_report(results: list, bah: dict, df_valid: pd.DataFrame):
    print("\n" + "=" * 66)
    print("  BACKTEST RESULTS  —  last 20% of data as test period")
    print("=" * 66)
    print(f"  Initial capital : ${INITIAL_CAPITAL:,.0f}")
    print(f"  Fee per side    : {FEE*100:.1f}%  (round-trip: {FEE*200:.1f}%)")
    print(f"  Horizon         : {label_config.horizon_candles} candles (4 h)")
    if len(df_valid) > 0:
        start = df_valid.iloc[0].get("open_time", "?")
        end   = df_valid.iloc[-1].get("open_time", "?")
        try:
            s = datetime.fromtimestamp(int(start)/1000, tz=timezone.utc).strftime("%Y-%m-%d")
            e = datetime.fromtimestamp(int(end)/1000,   tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"  Period          : {s} → {e}")
        except Exception:
            pass
    print()

    header = f"{'Metric':<30} " + "  ".join(f"{r['model']:>14}" for r in results)
    print(header)
    print("-" * len(header))

    def row(label, key, fmt=".2f", suffix=""):
        vals = "  ".join(f"{r[key]:>14{fmt}}{suffix}" for r in results)
        print(f"  {label:<28} {vals}")

    row("Macro-F1 (all preds)", "macro_f1",           ".4f", "")
    row("Trades taken",         "n_trades",           "d",   "")
    row("Total return",         "total_return_pct",   ".1f", "%")
    row("Win rate",             "win_rate_pct",        ".1f", "%")
    row("Avg return / trade",   "avg_trade_pct",       ".3f", "%")
    row("Sharpe ratio",         "sharpe",              ".2f", "")
    row("Max drawdown",         "max_drawdown_pct",    ".1f", "%")
    row("Final capital ($)",    "final_capital",       ",.0f", "")

    print("=" * 66)

    for r in results:
        bd = r["trade_breakdown"]
        print(f"\n  {r['model']} trade breakdown:")
        total = sum(bd.values())
        for label, count in [
            ("UP → LONG",          bd["UP"]),
            ("DOWN → SHORT",       bd["DOWN"]),
            ("FLAT → skip",        bd["FLAT"]),
            ("Low conf → skip",    bd.get("skipped_low_conf", 0)),
        ]:
            pct = count / total * 100 if total > 0 else 0
            print(f"    {label:<22} {count:5d}  ({pct:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Backtest trading strategy")
    parser.add_argument("--model", choices=["lstm", "quantum", "both"], default="both")
    args = parser.parse_args()

    from sklearn.metrics import f1_score

    results = []
    df_valid_ref = None

    if args.model in ("lstm", "both"):
        print("\nLoading data for LSTM")
        X_te, y_te, _, df_v, fe = load_data(feat_cols=LSTM_FEATURES_22)
        df_valid_ref = df_v
        print(f"Test samples: {len(X_te)}")
        try:
            probs, preds = predict_lstm(X_te, fe, LSTM_FEATURES_22)
            f1 = f1_score(y_te, preds, average="macro", zero_division=0)
            r = simulate(preds, probs, df_v, "LSTM", confidence_threshold=0.50)
            r["macro_f1"] = f1
            results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"LSTM skipped: {e}")

    if args.model in ("quantum", "both"):
        print("\nLoading data for VQC")
        X_te, y_te, _, df_v, _ = load_data(feat_cols=feature_config.feature_names)
        if df_valid_ref is None:
            df_valid_ref = df_v
        print(f" Test samples: {len(X_te)}")
        try:
            probs, preds = predict_quantum(X_te)
            f1 = f1_score(y_te, preds, average="macro", zero_division=0)
            r = simulate(preds, probs, df_v, "Quantum VQC", confidence_threshold=0.36)
            r["macro_f1"] = f1
            results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"Quantum skipped: {e}")

    df_valid = df_valid_ref
    bah = buy_and_hold(df_valid)
    print_report(results, bah, df_valid)


if __name__ == "__main__":
    main()
