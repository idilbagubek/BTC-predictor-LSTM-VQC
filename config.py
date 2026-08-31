"""
Configuration for BTC Price Prediction System
Stage 2: Enhanced LSTM with Attention, Focal Loss, Adaptive Epsilon
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "candles.db")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.pt")
CONFIG_PATH = os.path.join(MODEL_DIR, "training_config.pkl")
CALIBRATION_PATH = os.path.join(MODEL_DIR, "calibration.pkl")


@dataclass
class DataConfig:
    """Data collection and storage configuration"""
    symbol: str = "BTCUSDT"
    interval: str = "5m"
    exchange_api_base: str = "https://api.binance.com/api/v3"
    initial_fetch_limit: int = 1000
    table_name: str = "candles"


@dataclass
class LabelConfig:
    """Labeling configuration for 4-hour prediction with adaptive epsilon"""
    horizon_candles: int = 48
    
    # Static epsilon (fallback if adaptive disabled)
    epsilon: float = 0.0015
    epsilon_percentile: int = 35
    
    # Adaptive epsilon parameters
    use_adaptive_epsilon: bool = True
    k_range: Tuple[float, float] = (0.3, 1.5)
    k_default: float = 0.75
    k_search_steps: int = 15
    
    # Realized volatility window for adaptive epsilon
    vol_window: int = 48
    
    # Class labels
    classes: Tuple[str, ...] = ("DOWN", "FLAT", "UP")
    class_to_idx: dict = field(default_factory=lambda: {"DOWN": 0, "FLAT": 1, "UP": 2})
    idx_to_class: dict = field(default_factory=lambda: {0: "DOWN", 1: "FLAT", 2: "UP"})


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    window_size: int = 96
    
    # Technical indicator periods
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    ema_fast: int = 12
    ema_slow: int = 26
    momentum_period: int = 10
    volatility_period: int = 20
    volume_zscore_period: int = 20
    
    # Rolling regression for slope
    slope_period: int = 20
    
    # Volatility regime quantiles
    vol_regime_quantiles: Tuple[float, float] = (0.33, 0.67)
    
    # Feature list — ordered for correlation-aware quantum encoding.
    # Drop only pairs with corr > 0.9:
    #   close_open (1.00 with log_return_1), macd_signal (0.951 with macd),
    #   atr_norm (0.921 with realized_vol), ema_gap_slow (0.930 with ema_gap_fast).
    # Remaining 18 features are grouped so correlated features land in the same
    # VQC encoding round (6 qubits × 3 rounds) and get entangled by the CNOT ring.
    feature_names: List[str] = field(default_factory=lambda: [
        # Round 1 — RSI/EMA + MACD/trend cluster (corr 0.81–0.88)
        "rsi",
        "bb_pctb",
        "ema_gap_fast",
        "macd",
        "macd_hist",
        "slope",
        # Round 2 — volatility + return/range cluster (corr 0.80–0.83)
        "momentum",
        "realized_vol",
        "bb_width",
        "vol_regime",
        "log_return_1",
        "high_low_range",
        # Round 3 — volume pair + time cycles
        "log_volume",
        "volume_zscore",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ])


@dataclass
class ModelConfig:
    """LSTM model configuration with attention support"""
    # Architecture
    input_size: int = 18
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.30
    num_classes: int = 3
    
    # Attention configuration
    use_attention: bool = True
    attention_variant: str = "qk_scaled_dot"
    
    # Training
    learning_rate: float = 0.0005
    batch_size: int = 128
    max_epochs: int = 80
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 0.5
    weight_decay: float = 1e-4
    
    # Loss function
    use_focal_loss: bool = True
    focal_gamma: float = 1.0
    focal_alpha: Optional[List[float]] = None
    
    # Walk-forward split ratios (fallback for non-CV)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class CVConfig:
    """Cross-validation configuration"""
    n_splits: int = 3
    embargo_min_gap: int = 48
    validation_min_samples: int = 800
    use_expanding_window: bool = True
    
    # Memory management
    use_disk_backend: bool = True  # Auto-use SSD when RAM is insufficient
    ram_threshold_gb: float = 4.0  # Use disk if data exceeds this size


@dataclass 
class InferenceConfig:
    """Live inference configuration with calibration support"""
    min_confidence: float = 0.4
    confidence_margin: float = 0.15
    
    # Temperature scaling (learned during training)
    temperature: float = 1.0
    
    # Action mapping
    action_map: dict = field(default_factory=lambda: {
        "UP": "BUY",
        "DOWN": "SELL",
        "FLAT": "HOLD"
    })


@dataclass
class QuantumConfig:
    """Variational Quantum Classifier — Re-uploading architecture"""
    use_quantum: bool = False

    # Circuit architecture
    n_data_qubits: int = 6          # data encoding qubits
    # 3 readout qubits added automatically → total = 9 qubits
    # encoding rounds = ceil(18 / 6) = 3  (calculated in bridge)
    # Features are ordered in groups of 6 so each round encodes one
    # correlation cluster; the CNOT ring then entangles those correlated features.
    n_shots: int = 5
    spsa_estimates: int = 5         # average N SPSA estimates per gradient step

    # Training
    learning_rate: float = 0.004
    max_epochs: int = 200
    batch_size: int = 32
    early_stopping_patience: int = 30

    # Use only the last timestep of each sequence (22 features)
    use_last_timestep: bool = True


# Global config instances
data_config = DataConfig()
label_config = LabelConfig()
feature_config = FeatureConfig()
model_config = ModelConfig()
cv_config = CVConfig()
inference_config = InferenceConfig()
quantum_config = QuantumConfig()


def validate_config():
    """Validate configuration consistency at startup."""
    n_features = len(feature_config.feature_names)
    if model_config.input_size != n_features:
        raise AssertionError(
            f"Configuration mismatch: model_config.input_size ({model_config.input_size}) "
            f"!= len(feature_config.feature_names) ({n_features}). "
            f"Please update model_config.input_size to {n_features}."
        )
    
    total_ratio = model_config.train_ratio + model_config.val_ratio + model_config.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise AssertionError(
            f"Split ratios must sum to 1.0, got {total_ratio:.2f}"
        )
    
    if len(label_config.classes) != model_config.num_classes:
        raise AssertionError(
            f"Number of classes mismatch: label_config.classes has {len(label_config.classes)}, "
            f"model_config.num_classes is {model_config.num_classes}"
        )
    
    if model_config.attention_variant not in ["scaled_bilinear", "qk_scaled_dot", "none"]:
        raise AssertionError(
            f"Invalid attention_variant: {model_config.attention_variant}. "
            f"Must be 'scaled_bilinear', 'qk_scaled_dot', or 'none'."
        )
    
    if cv_config.embargo_min_gap < label_config.horizon_candles:
        raise AssertionError(
            f"embargo_min_gap ({cv_config.embargo_min_gap}) must be >= "
            f"horizon_candles ({label_config.horizon_candles}) to prevent label leakage."
        )
    
    if feature_config.window_size > 300 and data_config.interval == "1m":
        print("WARNING: Large window_size with 1m data may cause vanishing gradients. Consider 5m/15m candles.")

validate_config()
