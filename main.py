#!/usr/bin/env python3
"""
BTC Price Direction Predictor - Main Entry Point
=================================================

Enhanced pipeline for predicting BTC price direction over 4-hour horizons
using Attention LSTM with Focal Loss and Temperature Calibration.

Usage:
    python main.py collect    # Fetch historical data
    python main.py train      # Train the LSTM model
    python main.py predict    # Run live predictions
    python main.py all        # Run full pipeline
    python main.py status     # Check system status
"""

import sys
import os
import math
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_collect(args):
    """Fetch historical candle data"""
    from data_collector import init_database, fetch_historical_data, get_candle_count
    
    print("="*60)
    print("DATA COLLECTION")
    print("="*60)
    
    init_database()
    fetch_historical_data(days=args.days, verbose=True)
    
    count = get_candle_count()
    print(f"\nTotal candles in database: {count:,}")
    
    min_for_training = 10000
    if count >= min_for_training:
        print(f"[OK] Sufficient data for training ({count:,} >= {min_for_training:,})")
    else:
        needed_days = (min_for_training - count) // 1440 + 1
        print(f"[!] Recommend fetching more data. Need ~{needed_days} more days.")


def cmd_train(args):
    """Train the model (classical LSTM or hybrid quantum VQC)"""
    from data_collector import get_candle_count, init_database, load_candles_as_dataframe
    from config import model_config, cv_config, label_config, quantum_config

    use_quantum = getattr(args, "quantum", False)

    print("="*60)
    print(f"MODEL TRAINING ({'Hybrid Quantum VQC' if use_quantum else 'Classical LSTM'})")
    print("="*60)

    init_database()
    count = get_candle_count()

    if count < 2000:
        print(f"\nError: Only {count} candles available.")
        print("Run 'python main.py collect --days 30' first.")
        return

    if use_quantum:
        from trainer import QuantumTrainer
        from feature_engineer import FeatureEngineer

        print("\n[Quantum Config]")
        print(f"  Qubits : {quantum_config.n_data_qubits}")
        from config import feature_config
        print(f"  Features: {len(feature_config.feature_names)}")
        print(f"  Rounds : {math.ceil(len(feature_config.feature_names) / quantum_config.n_data_qubits)}")
        print(f"  Shots  : {quantum_config.n_shots}")
        print(f"  Epochs : {quantum_config.max_epochs}")

        df = load_candles_as_dataframe()
        fe = FeatureEngineer()
        df_features = fe.compute_features(df)
        from feature_engineer import build_sequences
        from config import label_config as lc
        labels = fe.compute_labels(df_features, use_adaptive=lc.use_adaptive_epsilon)
        X, y, timestamps = build_sequences(df_features, labels, include_current_candle=True)

        qt = QuantumTrainer()
        results = qt.train(X, y, timestamps)
        qt.save()

        print(f"\nAggregated Test F1: {results['test_metrics']['macro_f1']:.4f}")

    else:
        from trainer import main as train_main

        print("\n[Classical Config]")
        print(f"  Model: {'Attention LSTM' if model_config.use_attention else 'Basic LSTM'}")
        print(f"  Attention: {model_config.attention_variant}")
        print(f"  Hidden Size: {model_config.hidden_size}")
        print(f"  Layers: {model_config.num_layers}")
        print(f"  Loss: {'Focal Loss' if model_config.use_focal_loss else 'Cross Entropy'}")
        print(f"  CV Folds: {cv_config.n_splits}")
        print(f"  Embargo Gap: {cv_config.embargo_min_gap}")
        print(f"  Adaptive Epsilon: {label_config.use_adaptive_epsilon}")

        train_main()


def cmd_predict(args):
    """Run live predictions"""
    from live_predictor import main as predict_main
    
    sys.argv = ['live_predictor.py']
    if args.single:
        sys.argv.append('--single')
    if args.interval:
        sys.argv.extend(['--interval', str(args.interval)])
    
    predict_main()


def cmd_status(args):
    """Check system status"""
    from config import MODEL_PATH, SCALER_PATH, DB_PATH, CALIBRATION_PATH, model_config
    from data_collector import get_candle_count, get_oldest_candle_time, get_latest_candle_time, init_database
    import pickle
    
    print("="*60)
    print("SYSTEM STATUS")
    print("="*60)
    
    print("\n[Database]")
    db_exists = os.path.exists(DB_PATH)
    count = get_candle_count() if db_exists else 0
    
    if db_exists and count > 0:
        oldest = get_oldest_candle_time()
        latest = get_latest_candle_time()
        
        print(f"  Path: {DB_PATH}")
        print(f"  Candles: {count:,}")
        
        if oldest and latest:
            oldest_dt = datetime.fromtimestamp(oldest/1000, tz=timezone.utc)
            latest_dt = datetime.fromtimestamp(latest/1000, tz=timezone.utc)
            print(f"  Range: {oldest_dt.strftime('%Y-%m-%d %H:%M')} to {latest_dt.strftime('%Y-%m-%d %H:%M')}")
            
            age_minutes = (datetime.now(timezone.utc).timestamp() - latest/1000) / 60
            if age_minutes < 5:
                print(f"  Status: [OK] Fresh (updated {age_minutes:.0f}m ago)")
            else:
                print(f"  Status: [!] Stale (last update {age_minutes:.0f}m ago)")
    else:
        print(f"  Status: [X] Not found or empty")
        print(f"  Run: python main.py collect")
    
    print("\n[Model]")
    if os.path.exists(MODEL_PATH):
        import torch
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        
        print(f"  Path: {MODEL_PATH}")
        print(f"  Status: [OK] Trained")
        
        model_type = checkpoint.get('model_type', 'LSTMClassifier')
        config = checkpoint.get('config', {})
        print(f"  Type: {model_type}")
        
        if config:
            print(f"  Attention: {config.get('attention_variant', 'N/A')}")
            print(f"  Hidden: {config.get('hidden_size', 'N/A')}")
            print(f"  Layers: {config.get('num_layers', 'N/A')}")
        
        if 'training_history' in checkpoint and checkpoint['training_history']:
            last_epoch = checkpoint['training_history'][-1]
            print(f"  Last epoch: {last_epoch.get('epoch', '?')}")
            print(f"  Val F1: {last_epoch.get('val_macro_f1', 0):.4f}")
        
        if 'cv_results' in checkpoint and checkpoint['cv_results']:
            cv_f1s = [f['test_metrics']['macro_f1'] for f in checkpoint['cv_results']]
            print(f"  CV Folds: {len(cv_f1s)}")
            print(f"  CV Mean F1: {sum(cv_f1s)/len(cv_f1s):.4f}")
    else:
        print(f"  Status: [X] Not found")
        print(f"  Run: python main.py train")
    
    print("\n[Scaler]")
    if os.path.exists(SCALER_PATH):
        print(f"  Path: {SCALER_PATH}")
        print(f"  Status: [OK] Saved")
    else:
        print(f"  Status: [X] Not found")
    
    print("\n[Calibration]")
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH, 'rb') as f:
            cal_data = pickle.load(f)
        print(f"  Path: {CALIBRATION_PATH}")
        print(f"  Status: [OK] Saved")
        print(f"  Temperature: {cal_data.get('temperature', 'N/A'):.4f}")
        print(f"  Min Confidence: {cal_data.get('min_confidence', 'N/A'):.2f}")
        print(f"  Confidence Margin: {cal_data.get('confidence_margin', 'N/A'):.2f}")
        if 'k_value' in cal_data:
            print(f"  Adaptive k: {cal_data.get('k_value', 'N/A'):.4f}")
    else:
        print(f"  Status: [X] Not found (will use defaults)")
    
    print("\n[Readiness]")
    ready = all([
        db_exists and count >= 1000,
        os.path.exists(MODEL_PATH),
        os.path.exists(SCALER_PATH)
    ])
    
    if ready:
        print("  [OK] System ready for live predictions")
        print("  Run: python main.py predict")
    else:
        print("  [X] System not ready")
        print("  Required steps:")
        if not db_exists or count < 1000:
            print("    1. python main.py collect --days 30")
        if not os.path.exists(MODEL_PATH):
            print("    2. python main.py train")


def cmd_all(args):
    """Run full pipeline"""
    print("Running full pipeline...")
    print()
    
    args.days = getattr(args, 'days', 30)
    cmd_collect(args)
    print()
    
    cmd_train(args)
    print()
    
    cmd_status(args)
    print()
    
    print("Pipeline complete! Run 'python main.py predict' to start live predictions.")


def cmd_config(args):
    """Show current configuration"""
    from config import (
        data_config, label_config, feature_config, 
        model_config, cv_config, inference_config
    )
    
    print("="*60)
    print("CURRENT CONFIGURATION")
    print("="*60)
    
    print("\n[Data Config]")
    print(f"  Symbol: {data_config.symbol}")
    print(f"  Interval: {data_config.interval}")
    
    print("\n[Label Config]")
    print(f"  Horizon: {label_config.horizon_candles} candles")
    print(f"  Adaptive Epsilon: {label_config.use_adaptive_epsilon}")
    print(f"  k Range: {label_config.k_range}")
    
    print("\n[Feature Config]")
    print(f"  Window Size: {feature_config.window_size}")
    print(f"  Features: {len(feature_config.feature_names)}")
    
    print("\n[Model Config]")
    print(f"  Use Attention: {model_config.use_attention}")
    print(f"  Attention Variant: {model_config.attention_variant}")
    print(f"  Hidden Size: {model_config.hidden_size}")
    print(f"  Layers: {model_config.num_layers}")
    print(f"  Dropout: {model_config.dropout}")
    print(f"  Focal Loss: {model_config.use_focal_loss}")
    print(f"  Focal Gamma: {model_config.focal_gamma}")
    print(f"  Learning Rate: {model_config.learning_rate}")
    print(f"  Batch Size: {model_config.batch_size}")
    print(f"  Weight Decay: {model_config.weight_decay}")
    print(f"  Gradient Clip: {model_config.gradient_clip_norm}")
    
    print("\n[CV Config]")
    print(f"  Folds: {cv_config.n_splits}")
    print(f"  Embargo Gap: {cv_config.embargo_min_gap}")
    print(f"  Min Val Samples: {cv_config.validation_min_samples}")
    
    print("\n[Inference Config]")
    print(f"  Min Confidence: {inference_config.min_confidence}")
    print(f"  Confidence Margin: {inference_config.confidence_margin}")


def main():
    parser = argparse.ArgumentParser(
        description='BTC Price Direction Predictor (Enhanced)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py collect --days 30    Fetch 30 days of historical data
  python main.py train                Train the LSTM model with rolling CV
  python main.py predict              Start live predictions
  python main.py predict --single     Run single prediction
  python main.py status               Check system status
  python main.py config               Show current configuration
  python main.py all --days 30        Run full pipeline
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    collect_parser = subparsers.add_parser('collect', help='Fetch historical data')
    collect_parser.add_argument('--days', type=int, default=30,
                               help='Days of history to fetch (default: 30)')
    
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--quantum', action='store_true',
                              help='Use hybrid quantum VQC instead of classical LSTM')
    
    predict_parser = subparsers.add_parser('predict', help='Run live predictions')
    predict_parser.add_argument('--single', action='store_true',
                               help='Single prediction and exit')
    predict_parser.add_argument('--interval', type=int, default=15,
                               help='Update interval in MINUTES (default: 15)')
    
    status_parser = subparsers.add_parser('status', help='Check system status')
    
    config_parser = subparsers.add_parser('config', help='Show current configuration')
    
    all_parser = subparsers.add_parser('all', help='Run full pipeline')
    all_parser.add_argument('--days', type=int, default=30,
                           help='Days of history to fetch')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    commands = {
        'collect': cmd_collect,
        'train': cmd_train,
        'predict': cmd_predict,
        'status': cmd_status,
        'config': cmd_config,
        'all': cmd_all
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
