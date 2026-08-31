import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import pickle

from config import feature_config, label_config, SCALER_PATH


class FeatureEngineer:
    
    def __init__(self):
        self.scaler_params = None
        self.epsilon = label_config.epsilon
        self.k_value = label_config.k_default
        self.vol_quantiles = None
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        
        df['close_open'] = (df['close'] - df['open']) / df['open']
        
        df['log_volume'] = np.log(df['volume'] + 1)

        # rsi
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
                
        avg_gain = gain.ewm(alpha=1/feature_config.rsi_period, min_periods=feature_config.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/feature_config.rsi_period, min_periods=feature_config.rsi_period, adjust=False).mean()
                
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        df['rsi'] = rsi / 100.0

        # macd
        ema_fast = df['close'].ewm(span=feature_config.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=feature_config.macd_slow, adjust=False).mean()
                
        macd_line = (ema_fast - ema_slow) / df['close'] 
        signal_line = macd_line.ewm(span=feature_config.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
             
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram
        
        df['atr_norm'] = self._compute_atr(df, feature_config.atr_period) / df['close']
        
        bb_width, bb_pctb = self._compute_bollinger(
            df['close'],
            feature_config.bb_period,
            feature_config.bb_std
        )
        df['bb_width'] = bb_width
        df['bb_pctb'] = bb_pctb
        
        ema_fast = df['close'].ewm(span=feature_config.ema_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=feature_config.ema_slow, adjust=False).mean()
        df['ema_gap_fast'] = (df['close'] - ema_fast) / df['close']
        df['ema_gap_slow'] = (df['close'] - ema_slow) / df['close']
        
        df['momentum'] = df['close'].pct_change(feature_config.momentum_period)
        
        df['realized_vol'] = df['log_return_1'].rolling(
            window=feature_config.volatility_period
        ).std()
        
        vol_mean = df['volume'].rolling(window=feature_config.volume_zscore_period).mean()
        vol_std = df['volume'].rolling(window=feature_config.volume_zscore_period).std()
        df['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-8)
        
        if hasattr(df.index, 'hour'):
            hour = df.index.hour
            day_of_week = df.index.dayofweek
        else:
            hour = pd.Series(np.zeros(len(df)), index=df.index)
            day_of_week = pd.Series(np.zeros(len(df)), index=df.index)
        
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        df['slope'] = self._compute_slope(df['close'], feature_config.slope_period)
        
        df['vol_regime'] = self._compute_vol_regime(
            df['realized_vol'], 
            feature_config.vol_regime_quantiles
        )
        
        return df
    
    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average True Range"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr
    
    def _compute_bollinger(
        self, 
        prices: pd.Series, 
        period: int, 
        num_std: float
    ) -> Tuple[pd.Series, pd.Series]:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        width = (upper - lower) / sma
        pctb = (prices - lower) / (upper - lower + 1e-8)
        
        return width, pctb
    
    def _compute_slope(self, prices: pd.Series, period: int) -> pd.Series:
        log_prices = np.log(prices)
        
        def rolling_slope(window):
            if len(window) < period or np.any(np.isnan(window)):
                return np.nan
            x = np.arange(len(window))
            slope, _ = np.polyfit(x, window, 1)
            return slope
        
        slope = log_prices.rolling(window=period).apply(rolling_slope, raw=True)
        return slope
    
    def _compute_vol_regime(
        self, 
        realized_vol: pd.Series, 
        quantiles: Tuple[float, float]
    ) -> pd.Series:
        q_low, q_high = quantiles
        
        def assign_regime(val, q_low_val, q_high_val):
            if pd.isna(val):
                return np.nan
            if val <= q_low_val:
                return 0.0
            elif val <= q_high_val:
                return 0.5
            else:
                return 1.0
        
        expanding_q_low = realized_vol.expanding(min_periods=100).quantile(q_low)
        expanding_q_high = realized_vol.expanding(min_periods=100).quantile(q_high)
        
        regime = pd.Series(index=realized_vol.index, dtype=float)
        for i in range(len(realized_vol)):
            regime.iloc[i] = assign_regime(
                realized_vol.iloc[i],
                expanding_q_low.iloc[i],
                expanding_q_high.iloc[i]
            )
        
        return regime
    
    def compute_realized_vol_for_labels(
        self, 
        df: pd.DataFrame, 
        window: int = None
    ) -> pd.Series:
        window = window or label_config.vol_window
        log_returns = np.log(df['close'] / df['close'].shift(1))
        return log_returns.rolling(window=window).std()
    
    def compute_labels(
        self, 
        df: pd.DataFrame, 
        horizon: int = None,
        use_adaptive: bool = None,
        k: float = None
    ) -> pd.Series:
        horizon = horizon or label_config.horizon_candles
        use_adaptive = use_adaptive if use_adaptive is not None else label_config.use_adaptive_epsilon
        k = k if k is not None else self.k_value
        
        future_close = df['close'].shift(-horizon)
        r_future = np.log(future_close / df['close'])
        
        labels = pd.Series(index=df.index, dtype=float)
        
        if use_adaptive:
            sigma = self.compute_realized_vol_for_labels(df)
            epsilon_t = k * sigma * np.sqrt(horizon)
            
            labels[r_future < -epsilon_t] = 0
            labels[(r_future >= -epsilon_t) & (r_future <= epsilon_t)] = 1
            labels[r_future > epsilon_t] = 2
        else:
            labels[r_future < -self.epsilon] = 0
            labels[r_future.abs() <= self.epsilon] = 1
            labels[r_future > self.epsilon] = 2
        
        return labels
    
    def tune_epsilon(self, df: pd.DataFrame, percentile: int = None) -> float:
        percentile = percentile or label_config.epsilon_percentile
        horizon = label_config.horizon_candles
        
        future_close = df['close'].shift(-horizon)
        r_future = np.log(future_close / df['close'])
        
        abs_returns = r_future.abs().dropna()
        epsilon = np.percentile(abs_returns, percentile)
        epsilon = max(epsilon, 0.001)
        
        self.epsilon = epsilon
        return epsilon
    
    def tune_k_adaptive(
        self, 
        df: pd.DataFrame, 
        target_flat_ratio: float = 0.33
    ) -> float:
        k_min, k_max = label_config.k_range
        k_steps = label_config.k_search_steps
        
        best_k = label_config.k_default
        best_diff = float('inf')
        
        for k in np.linspace(k_min, k_max, k_steps):
            labels = self.compute_labels(df, use_adaptive=True, k=k)
            valid_labels = labels.dropna()
            
            if len(valid_labels) == 0:
                continue
            
            flat_ratio = (valid_labels == 1).sum() / len(valid_labels)
            diff = abs(flat_ratio - target_flat_ratio)
            
            if diff < best_diff:
                best_diff = diff
                best_k = k
        
        self.k_value = best_k
        return best_k
    
    def fit_scaler(self, df: pd.DataFrame) -> None:
        feature_cols = feature_config.feature_names
        
        missing = [f for f in feature_cols if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        self.scaler_params = {
            'mean': df[feature_cols].mean().to_dict(),
            'std': df[feature_cols].std().to_dict()
        }
        
        for col in feature_cols:
            if self.scaler_params['std'][col] < 1e-8:
                self.scaler_params['std'][col] = 1.0
    
    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_params is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        df = df.copy()
        feature_cols = feature_config.feature_names
        
        for col in feature_cols:
            mean = self.scaler_params['mean'][col]
            std = self.scaler_params['std'][col]
            df[col] = (df[col] - mean) / std
        
        return df
    
    def save_scaler(self, path: str = None) -> None:
        path = path or SCALER_PATH
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler_params': self.scaler_params,
                'epsilon': self.epsilon,
                'k_value': self.k_value,
                'vol_quantiles': self.vol_quantiles
            }, f)
        print(f"Scaler saved to {path}")
    
    def load_scaler(self, path: str = None) -> None:
        path = path or SCALER_PATH
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.scaler_params = data['scaler_params']
            self.epsilon = data.get('epsilon', label_config.epsilon)
            self.k_value = data.get('k_value', label_config.k_default)
            self.vol_quantiles = data.get('vol_quantiles', None)
        print(f"Scaler loaded from {path}")


def build_sequences(
    df: pd.DataFrame,
    labels: pd.Series,
    window_size: int = None,
    interval_minutes: int = 5,
    check_continuity: bool = True,
    include_current_candle: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_size = window_size or feature_config.window_size
    feature_cols = feature_config.feature_names
    
    features = df[feature_cols].values
    label_values = labels.values
    timestamps = df.index.values if hasattr(df.index, 'values') else np.arange(len(df))
    
    expected_delta = pd.Timedelta(minutes=interval_minutes * (window_size - 1))
    max_allowed_delta = expected_delta * 1.1
    
    sequences = []
    seq_labels = []
    seq_times = []
    skipped_gaps = 0
    skipped_nan = 0
    
    start_idx = window_size - 1 if include_current_candle else window_size
    
    for i in range(start_idx, len(df)):
        if pd.isna(label_values[i]):
            continue
        
        if include_current_candle:
            window = features[i - window_size + 1:i + 1]
        else:
            window = features[i - window_size:i]
        
        if np.any(np.isnan(window)):
            skipped_nan += 1
            continue
        
        if check_continuity:
            try:
                if include_current_candle:
                    t_start = pd.Timestamp(timestamps[i - window_size + 1])
                    t_end = pd.Timestamp(timestamps[i])
                else:
                    t_start = pd.Timestamp(timestamps[i - window_size])
                    t_end = pd.Timestamp(timestamps[i - 1])
                
                actual_delta = t_end - t_start
                
                if actual_delta > max_allowed_delta:
                    skipped_gaps += 1
                    continue
            except (TypeError, ValueError):
                pass
        
        sequences.append(window)
        seq_labels.append(int(label_values[i]))
        seq_times.append(timestamps[i])
    
    if skipped_gaps > 0:
        print(f"  Skipped {skipped_gaps} sequences due to timestamp gaps")
    if skipped_nan > 0:
        print(f"  Skipped {skipped_nan} sequences due to NaN features")
    
    X = np.array(sequences, dtype=np.float32)
    y = np.array(seq_labels, dtype=np.int64)
    times = np.array(seq_times)
    
    return X, y, times


def get_class_distribution(y: np.ndarray) -> dict:
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    dist = {}
    for cls, count in zip(unique, counts):
        label = label_config.idx_to_class[int(cls)]
        dist[label] = {
            'count': int(count),
            'percentage': round(100 * count / total, 1)
        }
    
    return dist


if __name__ == "__main__":
    from data_collector import load_candles_as_dataframe, init_database, fetch_historical_data
    
    print("Loading candle data...")
    df = load_candles_as_dataframe(limit=3000)
    
    if len(df) < 1000:
        print("Not enough data, fetching more...")
        init_database()
        fetch_historical_data(days=5)
        df = load_candles_as_dataframe(limit=3000)
    
    print(f"Loaded {len(df)} candles")
    
    fe = FeatureEngineer()
    df_features = fe.compute_features(df)
    
    print("\nFeature columns:")
    print(df_features[feature_config.feature_names].tail())
    
    print("\nFeature NaN counts:")
    nan_counts = df_features[feature_config.feature_names].isna().sum()
    print(nan_counts[nan_counts > 0])
    
    epsilon = fe.tune_epsilon(df_features)
    print(f"\nStatic epsilon: {epsilon:.4f} ({epsilon*100:.2f}%)")
    
    k = fe.tune_k_adaptive(df_features, target_flat_ratio=0.33)
    print(f"Adaptive k: {k:.4f}")
    
    print("\nComparing labeling methods:")
    labels_static = fe.compute_labels(df_features, use_adaptive=False)
    labels_adaptive = fe.compute_labels(df_features, use_adaptive=True, k=k)
    
    print("Static epsilon labels:")
    for cls_idx, cls_name in label_config.idx_to_class.items():
        count = (labels_static == cls_idx).sum()
        pct = 100 * count / labels_static.dropna().shape[0]
        print(f"  {cls_name}: {count} ({pct:.1f}%)")
    
    print("\nAdaptive epsilon labels:")
    for cls_idx, cls_name in label_config.idx_to_class.items():
        count = (labels_adaptive == cls_idx).sum()
        pct = 100 * count / labels_adaptive.dropna().shape[0]
        print(f"  {cls_name}: {count} ({pct:.1f}%)")
    
    fe.fit_scaler(df_features.dropna())
    df_normalized = fe.transform_features(df_features)
    
    print("\nNormalized feature stats:")
    print(df_normalized[feature_config.feature_names].describe().loc[['mean', 'std']])
    
    print("\nBuilding sequences (with current candle fix)...")
    X, y, times = build_sequences(df_normalized, labels_adaptive, include_current_candle=True)
    print(f"Sequences: {X.shape}")
    print(f"Labels: {y.shape}")
    print(f"Class distribution: {get_class_distribution(y)}")
