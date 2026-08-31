import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, List
import signal
import pickle

from config import (
    model_config, label_config, feature_config, inference_config,
    MODEL_PATH, SCALER_PATH, CALIBRATION_PATH
)
from model import LSTMClassifier, AttentionLSTMClassifier, create_model
from feature_engineer import FeatureEngineer
from data_collector import (
    load_candles_as_dataframe, update_candles, 
    get_candle_count, init_database
)


class PredictionHistory:
    
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.history: List[Dict] = []
    
    def add(self, result: Dict):
        self.history.append({
            'timestamp': result['timestamp'],
            'prediction': result['prediction'],
            'action': result['action'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'].copy(),
            'price': result['current_price']
        })
        
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_previous(self) -> Optional[Dict]:
        if len(self.history) >= 2:
            return self.history[-2]
        return None
    
    def get_signal_change(self, current: Dict) -> Optional[str]:
        """Detect if signal changed from previous prediction."""
        prev = self.get_previous()
        if prev is None:
            return None
        
        prev_action = prev['action'].split()[0]
        curr_action = current['action'].split()[0]
        
        if prev_action != curr_action:
            return f"{prev_action} -> {curr_action}"
        return None
    
    def get_confidence_trend(self, current: Dict) -> str:
        """Calculate confidence trend over recent predictions."""
        if len(self.history) < 2:
            return "->"
        
        recent = self.history[-3:] if len(self.history) >= 3 else self.history
        avg_conf = np.mean([h['confidence'] for h in recent])
        
        diff = current['confidence'] - avg_conf
        
        if diff > 0.10:
            return "^^"
        elif diff > 0.03:
            return "^"
        elif diff < -0.10:
            return "vv"
        elif diff < -0.03:
            return "v"
        else:
            return "->"
    
    def get_prediction_streak(self, current: Dict) -> int:
        """Count consecutive predictions of the same class."""
        if not self.history:
            return 1
        
        streak = 1
        current_pred = current['prediction']
        
        for h in reversed(self.history):
            if h['prediction'] == current_pred:
                streak += 1
            else:
                break
        
        return streak
    
    def get_price_change(self, current: Dict) -> Optional[float]:
        """Calculate price change since first prediction in history."""
        if not self.history:
            return None
        
        first_price = self.history[0]['price']
        current_price = current['current_price']
        
        return ((current_price - first_price) / first_price) * 100


class LivePredictor:
    
    def __init__(self, device: str = None, use_quantum: bool = False):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.use_quantum = use_quantum
        self.quantum_bridge = None
        self.quantum_mean = None
        self.quantum_std  = None
        self.feature_engineer = FeatureEngineer()
        self.last_prediction_time = None
        self.history = PredictionHistory()

        self.temperature = 1.0
        self.min_confidence = inference_config.min_confidence
        self.confidence_margin = inference_config.confidence_margin
        
    def load_model(self, model_path: str = None) -> bool:
        """Load trained model, scaler, and calibration."""
        model_path = model_path or MODEL_PATH
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train the model first using: python trainer.py")
            return False
        
        if not os.path.exists(SCALER_PATH):
            print(f"Error: Scaler not found at {SCALER_PATH}")
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            model_type = checkpoint.get('model_type', 'LSTMClassifier')
            config = checkpoint.get('config', {})

            # Detect input_size from saved weights so old checkpoints load correctly
            saved_input_size = (
                checkpoint['model_state_dict']['lstm.weight_ih_l0'].shape[1]
            )
            orig_input_size = model_config.input_size
            model_config.input_size = saved_input_size

            if model_type == 'AttentionLSTMClassifier' or config.get('use_attention', False):
                self.model = AttentionLSTMClassifier(
                    attention_variant=config.get('attention_variant', model_config.attention_variant)
                ).to(self.device)
                print(f"Loaded AttentionLSTMClassifier ({config.get('attention_variant', 'default')}, input_size={saved_input_size})")
            else:
                self.model = LSTMClassifier().to(self.device)
                print(f"Loaded LSTMClassifier (input_size={saved_input_size})")

            model_config.input_size = orig_input_size  # restore

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.feature_engineer.load_scaler()
            
            self.load_calibration()
            
            print(f"Model loaded successfully (device: {self.device})")
            print(f"Temperature: {self.temperature:.4f}")
            print(f"Min Confidence: {self.min_confidence:.2f}")
            print(f"Confidence Margin: {self.confidence_margin:.2f}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_calibration(self, path: str = None) -> None:
        """Load calibration parameters from training."""
        path = path or CALIBRATION_PATH
        
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.temperature = data.get('temperature', 1.0)
                self.min_confidence = data.get('min_confidence', inference_config.min_confidence)
                self.confidence_margin = data.get('confidence_margin', inference_config.confidence_margin)
                
                if 'k_value' in data:
                    self.feature_engineer.k_value = data['k_value']
                if 'epsilon' in data:
                    self.feature_engineer.epsilon = data['epsilon']
            
            print(f"Calibration loaded from {path}")
        else:
            print(f"Warning: Calibration file not found at {path}, using defaults")
    
    def load_quantum_model(self) -> bool:
        """Load trained Quantum VQC parameters and scaler."""
        from quantum_bridge import QuantumVQCBridge
        q_model  = os.path.join(os.path.dirname(MODEL_PATH), "quantum_params.pkl")
        q_scaler = os.path.join(os.path.dirname(MODEL_PATH), "quantum_scaler.pkl")

        if not os.path.exists(q_model):
            print(f"Error: quantum_params.pkl not found. Run: python main.py train --quantum")
            return False

        self.quantum_bridge = QuantumVQCBridge.load(q_model)

        if os.path.exists(q_scaler):
            with open(q_scaler, "rb") as f:
                sc = pickle.load(f)
            self.quantum_mean = sc["mean"]
            self.quantum_std  = sc["std"]

        print(f"Quantum model loaded  "
              f"(params={self.quantum_bridge.total_params}, "
              f"qubits={self.quantum_bridge.n_total}, "
              f"rounds={self.quantum_bridge.n_rounds})")
        return True

    def prepare_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        window_size = feature_config.window_size
        
        if len(df) < window_size + 50:
            print(f"Insufficient data: need {window_size + 50} candles, have {len(df)}")
            return None
        
        df_features = self.feature_engineer.compute_features(df)

        if self.use_quantum:
            # Quantum only needs the last timestep — raw features, scaler applied in predict()
            feat_cols = feature_config.feature_names
            last_row = df_features[feat_cols].dropna().iloc[-1].values.astype(np.float32)
            return last_row.reshape(1, -1)

        df_normalized = self.feature_engineer.transform_features(df_features)

        # Use feature list that matches the loaded model's input_size
        n_model_features = self.model.lstm.input_size if self.model else len(feature_config.feature_names)
        all_22 = [
            "log_return_1", "high_low_range", "close_open", "log_volume",
            "rsi", "macd", "macd_signal", "macd_hist",
            "atr_norm", "bb_width", "bb_pctb", "ema_gap_fast", "ema_gap_slow",
            "momentum", "realized_vol", "volume_zscore",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "slope", "vol_regime",
        ]
        feature_cols = all_22 if n_model_features == 22 else feature_config.feature_names
        features = df_normalized[feature_cols].iloc[-window_size:].values

        if np.any(np.isnan(features)):
            nan_counts = np.isnan(features).sum(axis=0)
            nan_features = [feature_cols[i] for i, c in enumerate(nan_counts) if c > 0]
            print(f"Warning: NaN in features: {nan_features}")
            features = np.nan_to_num(features, nan=0.0)

        features = features.reshape(1, window_size, -1).astype(np.float32)
        return features
    
    def predict(self, features: np.ndarray) -> Dict:
        if self.use_quantum:
            flat = features.flatten()
            if self.quantum_mean is not None:
                flat = (flat - self.quantum_mean) / self.quantum_std
            probs = self.quantum_bridge.predict_proba(flat)
        else:
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32).to(self.device)
                logits = self.model(x)[0].cpu().numpy()
                scaled_logits = logits / self.temperature
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
                probs = exp_logits / exp_logits.sum()
        
        pred_idx = np.argmax(probs)
        pred_class = label_config.idx_to_class[pred_idx]
        
        sorted_probs = np.sort(probs)[::-1]
        max_prob = sorted_probs[0]
        margin = sorted_probs[0] - sorted_probs[1]
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        
        action = self._determine_action(pred_class, max_prob, margin)
        
        return {
            'probabilities': {
                'DOWN': float(probs[0]),
                'FLAT': float(probs[1]),
                'UP': float(probs[2])
            },
            'prediction': pred_class,
            'confidence': float(max_prob),
            'margin': float(margin),
            'entropy': float(entropy),
            'action': action,
            'raw_logits': logits.tolist() if not self.use_quantum else probs.tolist(),
            'temperature': self.temperature
        }
    
    def _determine_action(
        self, 
        prediction: str, 
        confidence: float, 
        margin: float
    ) -> str:
        """Map prediction to actionable signal with calibrated thresholds."""
        if confidence < self.min_confidence:
            return "HOLD (low confidence)"
        
        if margin < self.confidence_margin:
            return "HOLD (uncertain)"
        
        base_action = inference_config.action_map.get(prediction, "HOLD")
        return base_action
    
    def run_single_prediction(self) -> Optional[Dict]:
        """Run a single prediction using latest data."""
        new_candles = update_candles()
        
        df = load_candles_as_dataframe(limit=feature_config.window_size + 100)
        
        if df.empty:
            print("No candle data available")
            return None
        
        features = self.prepare_features(df)
        if features is None:
            return None
        
        result = self.predict(features)
        
        result['timestamp'] = datetime.now(timezone.utc).isoformat()
        result['current_price'] = float(df['close'].iloc[-1])
        result['candle_time'] = str(df.index[-1])
        
        result['signal_change'] = self.history.get_signal_change(result)
        result['confidence_trend'] = self.history.get_confidence_trend(result)
        result['prediction_streak'] = self.history.get_prediction_streak(result)
        result['session_price_change'] = self.history.get_price_change(result)
        
        self.history.add(result)
        
        return result


class TerminalDisplay:
    """
    Formatted terminal output for predictions.
    """
    
    COLORS = {
        'GREEN': '\033[92m',
        'RED': '\033[91m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'CYAN': '\033[96m',
        'WHITE': '\033[97m',
        'BOLD': '\033[1m',
        'END': '\033[0m',
        'BG_GREEN': '\033[42m',
        'BG_RED': '\033[41m',
        'BG_YELLOW': '\033[43m',
    }
    
    _windows_initialized = False
    
    @classmethod
    def _init_windows(cls):
        """Enable ANSI colors on Windows"""
        if not cls._windows_initialized and os.name == 'nt':
            os.system('')
            cls._windows_initialized = True
    
    @classmethod
    def clear_screen(cls):
        """Clear terminal screen"""
        cls._init_windows()
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @classmethod
    def color(cls, text: str, color: str) -> str:
        """Apply color to text"""
        return f"{cls.COLORS.get(color, '')}{text}{cls.COLORS['END']}"
    
    @classmethod
    def print_header(cls, update_interval: int = 15):
        """Print application header"""
        print(cls.color("="*60, 'CYAN'))
        print(cls.color("  BTC Price Direction Predictor (Enhanced)", 'BOLD'))
        print(cls.color(f"  4-Hour Horizon | Updates every {update_interval} min", 'CYAN'))
        print(cls.color("  Temperature-Calibrated Probabilities", 'CYAN'))
        print(cls.color("="*60, 'CYAN'))
    
    @classmethod
    def print_signal_change_alert(cls, change: str):
        """Print prominent signal change alert"""
        print()
        print(cls.color("!" * 60, 'YELLOW'))
        print(cls.color(f"  SIGNAL CHANGE: {change}", 'BOLD'))
        print(cls.color("!" * 60, 'YELLOW'))
    
    @classmethod
    def print_prediction(cls, result: Dict, show_extended: bool = True):
        """Print formatted prediction result"""
        probs = result['probabilities']
        
        action = result['action']
        if 'BUY' in action:
            action_color = 'GREEN'
        elif 'SELL' in action:
            action_color = 'RED'
        else:
            action_color = 'YELLOW'
        
        pred = result['prediction']
        if pred == 'UP':
            pred_color = 'GREEN'
        elif pred == 'DOWN':
            pred_color = 'RED'
        else:
            pred_color = 'YELLOW'
        
        if result.get('signal_change'):
            cls.print_signal_change_alert(result['signal_change'])
        
        print(f"\n{cls.color('Timestamp:', 'WHITE')} {result['timestamp']}")
        print(f"{cls.color('Candle:', 'WHITE')}   {result['candle_time']}")
        print(f"{cls.color('Price:', 'WHITE')}     ${result['current_price']:,.2f}")
        
        if show_extended and result.get('session_price_change') is not None:
            price_chg = result['session_price_change']
            chg_color = 'GREEN' if price_chg >= 0 else 'RED'
            print(f"{cls.color('Session:', 'WHITE')}   {cls.color(f'{price_chg:+.2f}%', chg_color)}")
        
        print(f"\n{cls.color('-'*40, 'CYAN')}")
        print(cls.color("PROBABILITIES (Calibrated)", 'BOLD'))
        print(f"  DOWN:  {cls._prob_bar(probs['DOWN'], 'RED')}")
        print(f"  FLAT:  {cls._prob_bar(probs['FLAT'], 'YELLOW')}")
        print(f"  UP:    {cls._prob_bar(probs['UP'], 'GREEN')}")
        
        print(f"\n{cls.color('-'*40, 'CYAN')}")
        print(f"{cls.color('Prediction:', 'WHITE')} {cls.color(pred, pred_color)}", end='')
        
        streak = result.get('prediction_streak', 1)
        if streak > 1:
            print(f"  {cls.color(f'(x{streak})', 'CYAN')}", end='')
        print()
        
        trend = result.get('confidence_trend', '->')
        trend_color = 'GREEN' if '^' in trend else ('RED' if 'v' in trend else 'WHITE')
        print(f"{cls.color('Confidence:', 'WHITE')} {result['confidence']*100:.1f}% {cls.color(trend, trend_color)}")
        print(f"{cls.color('Margin:', 'WHITE')}     {result['margin']*100:.1f}%")
        
        if 'temperature' in result:
            print(f"{cls.color('Temperature:', 'WHITE')} {result['temperature']:.3f}")
        
        print(f"\n{cls.color('-'*40, 'CYAN')}")
        print(f"{cls.color('ACTION:', 'BOLD')} {cls.color(action, action_color)}")
        print(cls.color("="*60, 'CYAN'))
    
    @classmethod
    def _prob_bar(cls, prob: float, color: str, width: int = 20) -> str:
        """Create a visual probability bar"""
        filled = int(prob * width)
        bar = '#' * filled + '-' * (width - filled)
        return f"{cls.color(bar, color)} {prob*100:5.1f}%"
    
    @classmethod
    def print_status(cls, message: str, status: str = 'info'):
        """Print status message"""
        colors = {'info': 'CYAN', 'success': 'GREEN', 'error': 'RED', 'warning': 'YELLOW'}
        print(cls.color(f"[{status.upper()}] {message}", colors.get(status, 'WHITE')))
    
    @classmethod
    def print_waiting(cls, next_update: datetime, update_interval: int):
        """Print waiting status with countdown"""
        now = datetime.now(timezone.utc)
        remaining = (next_update - now).total_seconds()
        
        if remaining > 0:
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            print(f"\rNext update: {next_update.strftime('%H:%M:%S')} ({mins:02d}:{secs:02d}) ", 
                  end='', flush=True)


def get_next_update_time(interval_minutes: int = 15) -> datetime:
    """Calculate the next update time aligned to interval boundaries."""
    now = datetime.now(timezone.utc)
    
    minutes = now.minute
    next_minute = ((minutes // interval_minutes) + 1) * interval_minutes
    
    if next_minute >= 60:
        next_time = now.replace(minute=0, second=5, microsecond=0) + timedelta(hours=1)
    else:
        next_time = now.replace(minute=next_minute, second=5, microsecond=0)
    
    if next_time <= now:
        next_time += timedelta(minutes=interval_minutes)
    
    return next_time


def run_live_loop(
    predictor: LivePredictor,
    interval_minutes: int = 15,
    display: TerminalDisplay = None
):
    """
    Run continuous prediction loop with configurable interval.
    """
    display = display or TerminalDisplay()
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        print("\n\nShutting down...")
        running = False
    
    try:
        signal.signal(signal.SIGINT, signal_handler)
    except (OSError, AttributeError):
        pass
    
    print("\nStarting live prediction loop...")
    print(f"Update interval: {interval_minutes} minutes")
    print(f"Updates at: :00, :{interval_minutes:02d}, :{2*interval_minutes:02d}..." if interval_minutes <= 20 else "")
    print("Press Ctrl+C to stop\n")
    
    display.clear_screen()
    display.print_header(interval_minutes)
    
    result = predictor.run_single_prediction()
    if result:
        display.print_prediction(result)
    else:
        display.print_status("Failed to get prediction", "error")
    
    while running:
        try:
            next_update = get_next_update_time(interval_minutes)
            
            while running:
                now = datetime.now(timezone.utc)
                if now >= next_update:
                    break
                
                display.print_waiting(next_update, interval_minutes)
                time.sleep(1)
            
            if not running:
                break
            
            display.clear_screen()
            display.print_header(interval_minutes)
            
            result = predictor.run_single_prediction()
            
            if result:
                display.print_prediction(result)
                
                if result.get('signal_change') and os.name == 'nt':
                    try:
                        import winsound
                        winsound.Beep(1000, 200)
                    except:
                        pass
            else:
                display.print_status("Failed to get prediction", "error")
                
        except Exception as e:
            display.print_status(f"Error: {e}", "error")
            time.sleep(10)
    
    print("\nLive prediction stopped.")
    
    if predictor.history.history:
        print("\n" + "="*40)
        print("SESSION SUMMARY")
        print("="*40)
        print(f"Total predictions: {len(predictor.history.history)}")
        if predictor.history.history:
            first = predictor.history.history[0]
            last = predictor.history.history[-1]
            price_change = ((last['price'] - first['price']) / first['price']) * 100
            print(f"Price change: {price_change:+.2f}%")
            print(f"Started: {first['timestamp']}")
            print(f"Ended: {last['timestamp']}")


def run_single(predictor: LivePredictor):
    """Run a single prediction and exit"""
    display = TerminalDisplay()
    display.print_header()
    
    result = predictor.run_single_prediction()
    
    if result:
        display.print_prediction(result, show_extended=False)
        return result
    else:
        display.print_status("Failed to get prediction", "error")
        return None


def main():
    """Main entry point for live inference"""
    import argparse

    parser = argparse.ArgumentParser(description='BTC Price Direction Predictor')
    parser.add_argument('--quantum', action='store_true',
                        help='Use Quantum VQC instead of classical LSTM')
    parser.add_argument('--single', action='store_true',
                        help='Run single prediction and exit')
    parser.add_argument('--interval', type=int, default=15,
                        help='Update interval in MINUTES (default: 15)')
    args = parser.parse_args()

    init_database()

    count = get_candle_count()
    if count < feature_config.window_size:
        print(f"Insufficient data: {count} candles available, "
              f"need at least {feature_config.window_size}")
        print("Run: python main.py collect --days 30")
        sys.exit(1)

    predictor = LivePredictor(use_quantum=args.quantum)

    if args.quantum:
        if not predictor.load_quantum_model():
            sys.exit(1)
    else:
        if not predictor.load_model():
            print("\nPlease train the model first:")
            print("  python main.py train")
            sys.exit(1)

    if args.single:
        run_single(predictor)
    else:
        run_live_loop(predictor, interval_minutes=args.interval)


if __name__ == "__main__":
    main()
