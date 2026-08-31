import os
import gc
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)
from typing import Dict, Tuple, List
import pickle
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

from config import (
    model_config, label_config, feature_config, cv_config, quantum_config,
    MODEL_PATH, SCALER_PATH, CALIBRATION_PATH
)
from model import LSTMClassifier, AttentionLSTMClassifier, FocalLoss, count_parameters, create_model
from feature_engineer import FeatureEngineer, build_sequences, get_class_distribution
from data_collector import load_candles_as_dataframe



def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


class ExpandingWindowCV:
    """Expanding window CV with embargo/purge to prevent label leakage."""

    def __init__(self, n_splits=None, embargo_gap=None, val_size=None, test_size=None):
        self.n_splits = n_splits or cv_config.n_splits
        self.embargo_gap = embargo_gap or cv_config.embargo_min_gap
        self.val_size = val_size or cv_config.validation_min_samples
        self.test_size = test_size

    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        n_samples = len(X)
        if self.test_size is None:
            self.test_size = max(n_samples // (self.n_splits + 2), self.val_size)

        min_train = cv_config.validation_min_samples
        min_split = cv_config.validation_min_samples // 2
        splits = []

        for i in range(self.n_splits):
            test_end   = n_samples - (self.n_splits - 1 - i) * self.test_size
            test_start = test_end - self.test_size
            val_end    = test_start - self.embargo_gap
            val_start  = val_end - self.val_size
            train_end  = val_start - self.embargo_gap

            if train_end < self.test_size:
                continue

            train_idx = np.arange(0, train_end)
            val_idx   = np.arange(val_start, val_end)
            test_idx  = np.arange(test_start, test_end)

            if len(train_idx) > min_train and len(val_idx) > min_split and len(test_idx) > min_split:
                splits.append((train_idx, val_idx, test_idx))

        return splits


class TemperatureScaler:
    def __init__(self, initial_temp: float = 1.0):
        self.temperature = initial_temp

    def fit(self, logits: np.ndarray, labels: np.ndarray, lr=0.01, max_iter=100) -> float:
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        temp = nn.Parameter(torch.ones(1))
        opt = optim.LBFGS([temp], lr=lr, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            nn.CrossEntropyLoss()(logits_t / temp, labels_t).backward()
            return nn.CrossEntropyLoss()(logits_t / temp, labels_t)

        opt.step(closure)
        self.temperature = max(temp.item(), 0.1)
        return self.temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        s = logits / self.temperature
        e = np.exp(s - np.max(s, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


class ThresholdOptimizer:
    def __init__(self):
        self.best_min_confidence = 0.4
        self.best_confidence_margin = 0.15
        self.best_score = 0.0

    def optimize(self, probs: np.ndarray, labels: np.ndarray, metric='macro_f1') -> Tuple[float, float]:
        best_score, best_params = 0.0, (0.4, 0.15)
        for min_conf in np.arange(0.3, 0.6, 0.05):
            for margin in np.arange(0.05, 0.25, 0.05):
                preds = self._apply_thresholds(probs, min_conf, margin)
                score = f1_score(labels, preds, average='macro') if metric == 'macro_f1' else accuracy_score(labels, preds)
                if score > best_score:
                    best_score, best_params = score, (min_conf, margin)
        self.best_min_confidence, self.best_confidence_margin = best_params
        self.best_score = best_score
        return best_params

    def _apply_thresholds(self, probs: np.ndarray, min_conf: float, margin: float) -> np.ndarray:
        preds = np.argmax(probs, axis=1)
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        uncertain = (np.max(probs, axis=1) < min_conf) | (sorted_probs[:, 0] - sorted_probs[:, 1] < margin)
        preds[uncertain] = 1
        return preds


class Trainer:
    """
    Enhanced training pipeline with:
    - Focal Loss for class imbalance
    - Expanding Window CV with embargo/purge
    - Adaptive epsilon tuning per fold
    - Temperature calibration
    - Threshold optimization
    """

    def __init__(
        self,
        device: str = None,
        use_rolling_cv: bool = True,
        seed: int = 42,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.training_history = []
        self.cv_results = []
        self.class_weights = None
        self.use_rolling_cv = use_rolling_cv

        self.temp_scaler = TemperatureScaler()
        self.threshold_optimizer = ThresholdOptimizer()

        self.seed = set_seed(seed)

        print(f"Using device: {self.device}")
        print(f"Random seed: {self.seed}")
        print(f"Cross-validation: {'Expanding Window' if use_rolling_cv else 'Fixed Hold-Out'}")
        print(f"Model: {'Attention LSTM' if model_config.use_attention else 'Basic LSTM'}")
        print(f"Loss: {'Focal Loss' if model_config.use_focal_loss else 'Cross Entropy'}")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        tune_k_per_fold: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Prepare sequences from raw candle data."""
        print("Computing features...")
        df_features = self.feature_engineer.compute_features(df)

        train_end = int(len(df_features) * 0.6)
        train_df = df_features.iloc[:train_end]

        if label_config.use_adaptive_epsilon and not tune_k_per_fold:
            print("Tuning adaptive epsilon k...")
            k = self.feature_engineer.tune_k_adaptive(train_df, target_flat_ratio=0.33)
            print(f"  k: {k:.4f}")
        else:
            print("Tuning static epsilon...")
            epsilon = self.feature_engineer.tune_epsilon(train_df)
            print(f"  Epsilon: {epsilon:.4f} ({epsilon*100:.2f}%)")

        print("Computing labels...")
        labels = self.feature_engineer.compute_labels(
            df_features,
            use_adaptive=label_config.use_adaptive_epsilon
        )

        print("Building sequences...")
        X, y, timestamps = build_sequences(df_features, labels, include_current_candle=True)

        print(f"  Total samples: {len(y)}")
        print(f"  Sequence shape: {X.shape}")
        print(f"  Class distribution: {get_class_distribution(y)}")

        return X, y, timestamps, df_features
    
    def normalize_fold(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Normalize data within a fold using training statistics."""
        train_flat = X_train.reshape(-1, X_train.shape[-1])

        mean = np.nanmean(train_flat, axis=0)
        std = np.nanstd(train_flat, axis=0)
        std[std < 1e-8] = 1.0

        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        X_test_norm = (X_test - mean) / std

        scaler_params = {'mean': mean, 'std': std}

        return X_train_norm, X_val_norm, X_test_norm, scaler_params
    
    def compute_class_weights(self, y: np.ndarray) -> torch.Tensor:
        """Compute class weights for imbalanced data."""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        weights = total / (len(unique) * counts)
        weights = weights / weights.sum() * len(unique)
        
        self.class_weights = torch.FloatTensor(weights).to(self.device)
        
        return self.class_weights
    
    def create_criterion(self, class_weights: torch.Tensor) -> nn.Module:
        """Create loss function (Focal or CrossEntropy)."""
        if model_config.use_focal_loss:
            alpha = class_weights if model_config.focal_alpha is None else \
                    torch.tensor(model_config.focal_alpha).to(self.device)
            return FocalLoss(gamma=model_config.focal_gamma, alpha=alpha)
        else:
            return nn.CrossEntropyLoss(weight=class_weights)
    
    def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                model_config.gradient_clip_norm
            )
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        return_logits: bool = False
    ) -> Dict:
        """Evaluate model on data."""
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_logits = []
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                
                logits = model(batch_X)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.numpy())
                all_probs.extend(probs.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_logits = np.array(all_logits)
        
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        accuracy = accuracy_score(all_labels, all_preds)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        result = {
            'macro_f1': macro_f1,
            'balanced_accuracy': balanced_acc,
            'accuracy': accuracy,
            'per_class': {
                label_config.idx_to_class[i]: {
                    'precision': precision[i] if i < len(precision) else 0,
                    'recall': recall[i] if i < len(recall) else 0,
                    'f1': f1[i] if i < len(f1) else 0,
                    'support': support[i] if i < len(support) else 0
                }
                for i in range(model_config.num_classes)
            },
            'confusion_matrix': conf_matrix,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        if return_logits:
            result['logits'] = all_logits
        
        return result
    
    def train_single_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        fold_idx: int = 0,
        verbose: bool = True
    ) -> Tuple[nn.Module, Dict, List]:
        """Train model on a single fold."""
        model = create_model().to(self.device)

        if verbose and fold_idx == 0:
            print(f"    Model parameters: {count_parameters(model):,}")

        class_weights = self.compute_class_weights(y_train)

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )

        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        val_loader = DataLoader(val_dataset, batch_size=model_config.batch_size)
        
        criterion = self.create_criterion(class_weights)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=model_config.learning_rate,
            weight_decay=model_config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None
        fold_history = []
        
        for epoch in range(model_config.max_epochs):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            val_metrics = self.evaluate(model, val_loader)
            val_f1 = val_metrics['macro_f1']
            
            scheduler.step(val_f1)
            
            fold_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_macro_f1': val_f1,
                'val_balanced_acc': val_metrics['balanced_accuracy']
            })
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}: loss={train_loss:.4f}, val_F1={val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= model_config.early_stopping_patience:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch+1}")
                    break
        
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        final_val_metrics = self.evaluate(model, val_loader, return_logits=True)
        
        gc.collect()
        
        return model, final_val_metrics, fold_history
    
    def train_with_rolling_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """Train with Expanding Window CV with embargo/purge."""
        cv = ExpandingWindowCV()
        splits = cv.split(X)

        if len(splits) < 2:
            print("Warning: Not enough data for rolling CV, falling back to fixed split")
            return self.train_with_fixed_split(X, y, timestamps, verbose)

        print(f"\nExpanding Window CV with {len(splits)} folds:")
        print(f"  Embargo gap: {cv.embargo_gap} samples (horizon leakage prevention)")

        all_test_preds = []
        all_test_labels = []
        all_test_probs = []
        all_val_logits = []
        all_val_labels = []
        fold_metrics = []
        best_model = None
        best_fold_f1 = 0
        best_scaler_params = None

        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
            print(f"\n  Fold {fold_idx + 1}/{len(splits)}:")
            print(f"    Train: {len(train_idx)} samples (0 to {train_idx[-1]})")
            print(f"    Val: {len(val_idx)} samples ({val_idx[0]} to {val_idx[-1]})")
            print(f"    Test: {len(test_idx)} samples ({test_idx[0]} to {test_idx[-1]})")

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            X_train_norm, X_val_norm, X_test_norm, scaler_params = self.normalize_fold(
                X_train, X_val, X_test
            )

            model, val_metrics, fold_history = self.train_single_fold(
                X_train_norm, y_train,
                X_val_norm, y_val,
                fold_idx=fold_idx,
                verbose=verbose
            )

            test_dataset = TensorDataset(
                torch.tensor(X_test_norm, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long)
            )
            test_loader = DataLoader(test_dataset, batch_size=model_config.batch_size)
            test_metrics = self.evaluate(model, test_loader)

            print(f"    Val F1: {val_metrics['macro_f1']:.4f}, Test F1: {test_metrics['macro_f1']:.4f}")

            all_test_preds.extend(test_metrics['predictions'])
            all_test_labels.extend(test_metrics['labels'])
            all_test_probs.extend(test_metrics['probabilities'])

            if 'logits' in val_metrics:
                all_val_logits.extend(val_metrics['logits'])
                all_val_labels.extend(val_metrics['labels'])

            fold_metrics.append({
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'history': fold_history
            })

            if test_metrics['macro_f1'] > best_fold_f1:
                best_fold_f1 = test_metrics['macro_f1']
                best_model = deepcopy(model)
                best_scaler_params = scaler_params
                self.training_history = fold_history

            del X_train_norm, X_val_norm, X_test_norm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_test_preds = np.array(all_test_preds)
        all_test_labels = np.array(all_test_labels)
        all_test_probs = np.array(all_test_probs)

        if len(all_val_logits) > 0:
            print("\n  Calibrating temperature...")
            all_val_logits = np.array(all_val_logits)
            all_val_labels = np.array(all_val_labels)

            temp = self.temp_scaler.fit(all_val_logits, all_val_labels)
            print(f"    Optimal temperature: {temp:.4f}")

            cal_probs = self.temp_scaler.calibrate(all_val_logits)

            print("  Optimizing thresholds...")
            best_thresholds = self.threshold_optimizer.optimize(cal_probs, all_val_labels)
            print(f"    Best thresholds: min_conf={best_thresholds[0]:.2f}, margin={best_thresholds[1]:.2f}")
            print(f"    Threshold-optimized F1: {self.threshold_optimizer.best_score:.4f}")

        aggregated_metrics = self._compute_aggregated_metrics(all_test_labels, all_test_preds, all_test_probs)

        self.model = best_model
        if best_scaler_params:
            self.feature_engineer.scaler_params = {
                'mean': {name: best_scaler_params['mean'][i]
                        for i, name in enumerate(feature_config.feature_names)},
                'std': {name: best_scaler_params['std'][i]
                       for i, name in enumerate(feature_config.feature_names)}
            }

        self.cv_results = fold_metrics

        self._print_cv_summary(fold_metrics, aggregated_metrics)

        return {
            'test_metrics': aggregated_metrics,
            'fold_metrics': fold_metrics,
            'cv_summary': self._get_cv_summary(fold_metrics),
            'calibration': {
                'temperature': self.temp_scaler.temperature,
                'min_confidence': self.threshold_optimizer.best_min_confidence,
                'confidence_margin': self.threshold_optimizer.best_confidence_margin
            }
        }
    
    def _compute_aggregated_metrics(
        self, 
        labels: np.ndarray, 
        preds: np.ndarray, 
        probs: np.ndarray
    ) -> Dict:
        """Compute aggregated metrics across all folds."""
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )
        
        return {
            'macro_f1': f1_score(labels, preds, average='macro'),
            'balanced_accuracy': balanced_accuracy_score(labels, preds),
            'accuracy': accuracy_score(labels, preds),
            'confusion_matrix': confusion_matrix(labels, preds),
            'predictions': preds,
            'labels': labels,
            'probabilities': probs,
            'per_class': {
                label_config.idx_to_class[i]: {
                    'precision': precision[i] if i < len(precision) else 0,
                    'recall': recall[i] if i < len(recall) else 0,
                    'f1': f1[i] if i < len(f1) else 0,
                    'support': support[i] if i < len(support) else 0
                }
                for i in range(model_config.num_classes)
            }
        }
    
    def _get_cv_summary(self, fold_metrics: List[Dict]) -> Dict:
        """Get summary statistics from CV folds."""
        fold_f1s = [f['test_metrics']['macro_f1'] for f in fold_metrics]
        return {
            'mean_f1': np.mean(fold_f1s),
            'std_f1': np.std(fold_f1s),
            'min_f1': np.min(fold_f1s),
            'max_f1': np.max(fold_f1s)
        }
    
    def _print_cv_summary(self, fold_metrics: List[Dict], aggregated_metrics: Dict):
        """Print CV summary."""
        print("\n" + "="*60)
        print("EXPANDING WINDOW CV SUMMARY")
        print("="*60)
        fold_f1s = [f['test_metrics']['macro_f1'] for f in fold_metrics]
        print(f"  Per-fold Test F1: {[f'{f:.4f}' for f in fold_f1s]}")
        print(f"  Mean F1: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")
        print(f"  Aggregated F1: {aggregated_metrics['macro_f1']:.4f}")
    
    def train_with_fixed_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """Train with traditional fixed hold-out split."""
        n_samples = len(X)
        
        train_end = int(n_samples * model_config.train_ratio)
        val_end = int(n_samples * (model_config.train_ratio + model_config.val_ratio))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        X_train_norm, X_val_norm, X_test_norm, scaler_params = self.normalize_fold(
            X_train, X_val, X_test
        )
        
        self.feature_engineer.scaler_params = {
            'mean': {name: scaler_params['mean'][i] 
                    for i, name in enumerate(feature_config.feature_names)},
            'std': {name: scaler_params['std'][i] 
                   for i, name in enumerate(feature_config.feature_names)}
        }
        
        print(f"\nFixed hold-out split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        self.model, val_metrics, self.training_history = self.train_single_fold(
            X_train_norm, y_train,
            X_val_norm, y_val,
            verbose=verbose
        )
        
        print(f"\nBest validation macro-F1: {val_metrics['macro_f1']:.4f}")
        
        test_dataset = TensorDataset(
            torch.tensor(X_test_norm, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )
        test_loader = DataLoader(test_dataset, batch_size=model_config.batch_size)
        test_metrics = self.evaluate(self.model, test_loader, return_logits=True)
        
        if 'logits' in val_metrics:
            temp = self.temp_scaler.fit(val_metrics['logits'], val_metrics['labels'])
            print(f"Temperature: {temp:.4f}")
            
            cal_probs = self.temp_scaler.calibrate(val_metrics['logits'])
            self.threshold_optimizer.optimize(cal_probs, val_metrics['labels'])
        
        return {
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'training_history': self.training_history,
            'calibration': {
                'temperature': self.temp_scaler.temperature,
                'min_confidence': self.threshold_optimizer.best_min_confidence,
                'confidence_margin': self.threshold_optimizer.best_confidence_margin
            }
        }
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """Main training method."""
        if self.use_rolling_cv:
            return self.train_with_rolling_cv(X, y, timestamps, verbose=verbose)
        else:
            return self.train_with_fixed_split(X, y, timestamps, verbose=verbose)
    
    def save_model(self, path: str = None) -> None:
        """Save trained model, config, and calibration."""
        path = path or MODEL_PATH
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model.__class__.__name__,
            'class_weights': self.class_weights,
            'training_history': self.training_history,
            'cv_results': self.cv_results,
            'seed': self.seed,
            'config': {
                'use_attention': model_config.use_attention,
                'attention_variant': model_config.attention_variant,
                'hidden_size': model_config.hidden_size,
                'num_layers': model_config.num_layers
            }
        }, path)
        
        self.feature_engineer.save_scaler()
        self.save_calibration()
        
        print(f"Model saved to {path}")
    
    def save_calibration(self, path: str = None) -> None:
        """Save calibration parameters."""
        path = path or CALIBRATION_PATH
        
        with open(path, 'wb') as f:
            pickle.dump({
                'temperature': self.temp_scaler.temperature,
                'min_confidence': self.threshold_optimizer.best_min_confidence,
                'confidence_margin': self.threshold_optimizer.best_confidence_margin,
                'k_value': self.feature_engineer.k_value,
                'epsilon': self.feature_engineer.epsilon
            }, f)
        
        print(f"Calibration saved to {path}")
    
    def load_model(self, path: str = None) -> None:
        """Load trained model."""
        path = path or MODEL_PATH
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        model_type = checkpoint.get('model_type', 'LSTMClassifier')
        config = checkpoint.get('config', {})
        
        if model_type == 'AttentionLSTMClassifier' or config.get('use_attention', False):
            self.model = AttentionLSTMClassifier(
                attention_variant=config.get('attention_variant', model_config.attention_variant)
            ).to(self.device)
        else:
            self.model = LSTMClassifier().to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_weights = checkpoint.get('class_weights')
        self.training_history = checkpoint.get('training_history', [])
        self.cv_results = checkpoint.get('cv_results', [])
        
        self.feature_engineer.load_scaler()
        self.load_calibration()
        
        print(f"Model loaded from {path}")
    
    def load_calibration(self, path: str = None) -> None:
        """Load calibration parameters."""
        path = path or CALIBRATION_PATH
        
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.temp_scaler.temperature = data.get('temperature', 1.0)
                self.threshold_optimizer.best_min_confidence = data.get('min_confidence', 0.4)
                self.threshold_optimizer.best_confidence_margin = data.get('confidence_margin', 0.15)
                
                if 'k_value' in data:
                    self.feature_engineer.k_value = data['k_value']
                if 'epsilon' in data:
                    self.feature_engineer.epsilon = data['epsilon']
            
            print(f"Calibration loaded from {path}")


class BaselineModels:
    @staticmethod
    def _metrics(y_true, y_pred, name):
        return {
            'name': name,
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
        }

    @staticmethod
    def _compute_metrics(y_true, y_pred, name): return BaselineModels._metrics(y_true, y_pred, name)

    @staticmethod
    def always_flat(y):  return BaselineModels._metrics(y, np.ones_like(y),       "Always-FLAT")
    @staticmethod
    def always_up(y):    return BaselineModels._metrics(y, np.full_like(y, 2),    "Always-UP")
    @staticmethod
    def always_down(y):  return BaselineModels._metrics(y, np.zeros_like(y),      "Always-DOWN")

    @staticmethod
    def logistic_regression_last_step(X_train, y_train, X_test, y_test):
        Xtr, Xte = X_train[:, -1, :], X_test[:, -1, :]
        m = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
        m.fit(Xtr, y_train)
        y_pred = m.predict(Xte)
        return y_pred, BaselineModels._metrics(y_test, y_pred, "LogReg-LastStep")

    @staticmethod
    def mlp_last_step(X_train, y_train, X_test, y_test):
        Xtr, Xte = X_train[:, -1, :], X_test[:, -1, :]
        m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                          early_stopping=True, validation_fraction=0.1, random_state=42)
        m.fit(Xtr, y_train)
        y_pred = m.predict(Xte)
        return y_pred, BaselineModels._metrics(y_test, y_pred, "MLP-LastStep")


def run_baselines_cv(X: np.ndarray, y: np.ndarray, n_splits: int = None) -> List[Dict]:
    """
    Run baseline models using the same CV folds as LSTM for fair comparison.
    """
    n_splits = n_splits or cv_config.n_splits
    cv = ExpandingWindowCV(n_splits=n_splits)
    splits = cv.split(X)
    
    if len(splits) < 2:
        return run_baselines_simple(X, y)
    
    print(f"\nRunning baselines with {len(splits)}-fold CV (same as LSTM)...")
    
    logreg_all_preds = []
    logreg_all_labels = []
    mlp_all_preds = []
    mlp_all_labels = []
    naive_all_labels = []
    
    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        train_flat = X_train.reshape(-1, X_train.shape[-1])
        mean = np.nanmean(train_flat, axis=0)
        std = np.nanstd(train_flat, axis=0)
        std[std < 1e-8] = 1.0
        
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std
        
        logreg_preds, _ = BaselineModels.logistic_regression_last_step(
            X_train_norm, y_train, X_test_norm, y_test
        )
        mlp_preds, _ = BaselineModels.mlp_last_step(
            X_train_norm, y_train, X_test_norm, y_test
        )
        
        logreg_all_preds.extend(logreg_preds)
        logreg_all_labels.extend(y_test)
        mlp_all_preds.extend(mlp_preds)
        mlp_all_labels.extend(y_test)
        naive_all_labels.extend(y_test)
    
    results = []
    
    naive_labels = np.array(naive_all_labels)
    results.append(BaselineModels.always_flat(naive_labels))
    results.append(BaselineModels.always_up(naive_labels))
    results.append(BaselineModels.always_down(naive_labels))
    
    results.append(BaselineModels._compute_metrics(
        np.array(logreg_all_labels), 
        np.array(logreg_all_preds), 
        "LogReg-LastStep"
    ))
    results.append(BaselineModels._compute_metrics(
        np.array(mlp_all_labels), 
        np.array(mlp_all_preds), 
        "MLP-LastStep"
    ))
    
    return results


def run_baselines_simple(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7) -> List[Dict]:
    """Simple baseline evaluation with fixed split (fallback)."""
    n = len(X)
    train_end = int(n * train_ratio)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[train_end:], y[train_end:]
    
    print("\nRunning baseline comparisons (simple split)...")
    results = []
    results.append(BaselineModels.always_flat(y_test))
    results.append(BaselineModels.always_up(y_test))
    results.append(BaselineModels.always_down(y_test))
    
    _, logreg_metrics = BaselineModels.logistic_regression_last_step(X_train, y_train, X_test, y_test)
    results.append(logreg_metrics)
    
    _, mlp_metrics = BaselineModels.mlp_last_step(X_train, y_train, X_test, y_test)
    results.append(mlp_metrics)
    
    return results


def run_baselines(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7) -> List[Dict]:
    """Run all baseline models - uses CV if enough data."""
    if len(X) > 5000:
        return run_baselines_cv(X, y)
    else:
        return run_baselines_simple(X, y, train_ratio)


def print_results_comparison(lstm_metrics: Dict, baseline_results: List[Dict]) -> None:
    """Print formatted comparison of LSTM vs baselines."""
    print("\n" + "="*60)
    print("MODEL COMPARISON (Test Set)")
    print("="*60)
    
    print(f"\n{'Model':<25} {'Macro-F1':>10} {'Bal.Acc':>10} {'Acc':>10}")
    print("-"*60)
    
    print(f"{'LSTM (ours)':<25} {lstm_metrics['macro_f1']:>10.4f} "
          f"{lstm_metrics['balanced_accuracy']:>10.4f} {lstm_metrics['accuracy']:>10.4f}")
    
    print("-"*60)
    
    for baseline in baseline_results:
        print(f"{baseline['name']:<25} {baseline['macro_f1']:>10.4f} "
              f"{baseline['balanced_accuracy']:>10.4f} {baseline['accuracy']:>10.4f}")
    
    print("\n" + "="*60)
    print("LSTM Per-Class Metrics")
    print("="*60)
    
    print(f"\n{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-"*50)
    
    for cls, metrics in lstm_metrics['per_class'].items():
        print(f"{cls:<10} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['support']:>10}")
    
    print("\nConfusion Matrix:")
    print("Predicted ->  DOWN  FLAT    UP")
    for i, row in enumerate(lstm_metrics['confusion_matrix']):
        label = label_config.idx_to_class[i]
        print(f"  {label:<6}    {row[0]:>4}  {row[1]:>4}  {row[2]:>4}")


class QuantumTrainer:
    """
    Hybrid Quantum Trainer.

    Differences with LSTM:
      - Input: 18 features
      - Gradients: parameter-shift rule, not PyTorch autograd
      - Adam: implemented manually for numpy parameter arrays
      - Output: saved to models/quantum_params.pkl instead of lstm_model.pt
    """

    QUANTUM_MODEL_PATH = os.path.join(os.path.dirname(MODEL_PATH), "quantum_params.pkl")

    def __init__(self, seed: int = 42):
        from quantum_bridge import QuantumVQCBridge

        set_seed(seed)
        self.seed = seed
        self.feature_engineer = FeatureEngineer()
        self.bridge = QuantumVQCBridge(
            n_features=len(feature_config.feature_names),
            n_qubits=quantum_config.n_data_qubits,
            n_shots=quantum_config.n_shots,
        )
        self.temp_scaler = TemperatureScaler()
        self.threshold_optimizer = ThresholdOptimizer()
        self.cv_results = []

        print(f"[QuantumTrainer] n_data_qubits={quantum_config.n_data_qubits}, "
              f"n_rounds={self.bridge.n_rounds}, "
              f"n_shots={quantum_config.n_shots}")
        print(f"[QuantumTrainer] Total circuit params: {self.bridge.total_params}")
        print(f"[QuantumTrainer] Q# available: {self.bridge._qsharp_available}")

    def _setup_correlation_entanglement(self, X: np.ndarray) -> None:
        corr = np.corrcoef(X[:, -1, :].T)
        self.bridge.set_entanglement_from_correlations(corr, threshold=0.7)

    def _adam_init(self) -> dict:
        n = len(self.bridge.params)
        return {"m": np.zeros(n), "v": np.zeros(n), "t": 0,
                "lr": quantum_config.learning_rate, "b1": 0.9, "b2": 0.999, "eps": 1e-8}

    def _adam_step(self, adam: dict, grad: np.ndarray) -> None:
        adam["t"] += 1
        adam["m"] = adam["b1"] * adam["m"] + (1 - adam["b1"]) * grad
        adam["v"] = adam["b2"] * adam["v"] + (1 - adam["b2"]) * grad ** 2
        m_hat = adam["m"] / (1 - adam["b1"] ** adam["t"])
        v_hat = adam["v"] / (1 - adam["b2"] ** adam["t"])
        self.bridge.params -= adam["lr"] * m_hat / (np.sqrt(v_hat) + adam["eps"])

    @staticmethod
    def _extract_last_step(X: np.ndarray) -> np.ndarray:
        """[N, window, features] → [N, features] (last timestep only)."""
        if X.ndim == 3:
            return X[:, -1, :]
        return X


    def _train_epoch(self, X_flat: np.ndarray, y: np.ndarray, adam: dict) -> float:
        n = len(y)
        indices = np.random.permutation(n)
        total_loss = 0.0
        batches = 0

        n_batches = math.ceil(n / quantum_config.batch_size)
        for batch_num, start in enumerate(range(0, n, quantum_config.batch_size)):
            batch_idx = indices[start:start + quantum_config.batch_size]
            batch_grad = np.zeros_like(self.bridge.params)
            batch_loss = 0.0

            for i in batch_idx:
                features = X_flat[i]
                target_idx = int(y[i])
                target_one_hot = np.eye(3)[target_idx]

                probs = self.bridge.predict_proba(features)
                batch_loss += -np.log(probs[target_idx] + 1e-9)
                batch_grad += self.bridge.spsa_gradient(features, target_one_hot,
                                                         n_estimates=quantum_config.spsa_estimates)

            batch_grad /= len(batch_idx)
            self._adam_step(adam, batch_grad)

            total_loss += batch_loss / len(batch_idx)
            batches += 1
            print(f"\r    Batch {batch_num+1}/{n_batches} — loss={batch_loss/len(batch_idx):.4f}", end="", flush=True)

        return total_loss / max(batches, 1)



    def _optimize_class_scales(
        self, probs: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Grid-search per-class scale factors that maximise macro-F1 on validation.
        Scales are multiplied into softmax probs before argmax — this shifts the
        decision boundary toward under-predicted minority classes without retraining.
        """
        best_f1, best_scales = 0.0, np.ones(3)
        for s_down in np.arange(1.0, 3.1, 0.25):
            for s_flat in np.arange(0.3, 1.1, 0.1):
                for s_up in np.arange(1.0, 3.1, 0.25):
                    scales = np.array([s_down, s_flat, s_up])
                    preds = np.argmax(probs * scales, axis=1)
                    f1 = f1_score(labels, preds, average="macro", zero_division=0)
                    if f1 > best_f1:
                        best_f1, best_scales = f1, scales
        print(f"    [ClassScales] DOWN×{best_scales[0]:.2f}, FLAT×{best_scales[1]:.2f}, "
              f"UP×{best_scales[2]:.2f}  → val F1={best_f1:.4f}")
        return best_scales


    def _evaluate_flat(
        self, X_flat: np.ndarray, y: np.ndarray,
        class_scales: np.ndarray | None = None,
    ) -> dict:
        preds, probs_list = [], []
        for i in range(len(y)):
            p = self.bridge.predict_proba(X_flat[i])
            probs_list.append(p)

        probs = np.array(probs_list)
        labels = np.array(y)

        if class_scales is not None:
            preds = np.argmax(probs * class_scales, axis=1)
        else:
            preds = np.argmax(probs, axis=1)

        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        balanced_acc = balanced_accuracy_score(labels, preds)
        accuracy = accuracy_score(labels, preds)

        return {
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_acc,
            "accuracy": accuracy,
            "predictions": preds,
            "labels": labels,
            "probabilities": probs,
        }


    def _train_fold(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        fold_idx: int,
    ) -> dict:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-8] = 1.0
        X_train_n = (X_train - mean) / std
        X_val_n = (X_val - mean) / std

        rng = np.random.default_rng(self.seed + fold_idx)
        self.bridge.params = rng.uniform(-0.1, 0.1, size=self.bridge.total_params)
        adam = self._adam_init()

        best_val_f1 = 0.0
        best_params = self.bridge.params.copy()
        patience = 0

        for epoch in range(quantum_config.max_epochs):
            loss = self._train_epoch(X_train_n, y_train, adam)
            val_metrics = self._evaluate_flat(X_val_n, y_val)
            val_f1 = val_metrics["macro_f1"]

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}: loss={loss:.4f}, val_F1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_params = self.bridge.params.copy()
                patience = 0
            else:
                patience += 1
                if patience >= quantum_config.early_stopping_patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        self.bridge.params = best_params
        return val_metrics, mean, std


    def train(self, X: np.ndarray, y: np.ndarray, timestamps: np.ndarray) -> dict:
        X_flat = self._extract_last_step(X)

        self._setup_correlation_entanglement(X)

        cv = ExpandingWindowCV()
        splits = cv.split(X_flat)

        if len(splits) < 2:
            raise RuntimeError("Not enough data for expanding-window CV.")

        print(f"\n[QuantumTrainer] Expanding Window CV — {len(splits)} folds")
        print("  (Each sample runs 2 × n_params circuit evaluations for gradient)")

        all_test_preds, all_test_labels, all_test_probs = [], [], []
        fold_metrics = []
        best_val_f1 = 0.0
        best_params = None
        best_scaler = None

        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
            print(f"\n  Fold {fold_idx+1}/{len(splits)}: "
                  f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

            X_tr, y_tr = X_flat[train_idx], y[train_idx]
            X_v,  y_v  = X_flat[val_idx],   y[val_idx]
            X_te, y_te = X_flat[test_idx],  y[test_idx]

            val_metrics, mean, std = self._train_fold(X_tr, y_tr, X_v, y_v, fold_idx)

            X_v_n = (X_flat[val_idx] - mean) / std
            val_probs = np.array([self.bridge.predict_proba(X_v_n[i]) for i in range(len(y_v))])
            class_scales = self._optimize_class_scales(val_probs, y_v)

            X_te_n = (X_te - mean) / std
            test_metrics = self._evaluate_flat(X_te_n, y_te, class_scales=class_scales)

            print(f"Val F1: {val_metrics['macro_f1']:.4f}, "
                  f"Test F1: {test_metrics['macro_f1']:.4f}")

            all_test_preds.extend(test_metrics["predictions"])
            all_test_labels.extend(test_metrics["labels"])
            all_test_probs.extend(test_metrics["probabilities"])

            fold_metrics.append({
                "fold": fold_idx + 1,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            })

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_params = self.bridge.params.copy()
                best_scaler = {"mean": mean, "std": std}

        if best_params is not None:
            self.bridge.params = best_params

        all_test_labels = np.array(all_test_labels)
        all_test_preds  = np.array(all_test_preds)
        all_test_probs  = np.array(all_test_probs)

        aggregated = {
            "macro_f1": f1_score(all_test_labels, all_test_preds, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(all_test_labels, all_test_preds),
            "accuracy": accuracy_score(all_test_labels, all_test_preds),
            "predictions": all_test_preds,
            "labels": all_test_labels,
            "probabilities": all_test_probs,
        }

        self.cv_results = fold_metrics
        self._best_scaler = best_scaler

        fold_f1s = [f["test_metrics"]["macro_f1"] for f in fold_metrics]
        print(f"\n{'='*60}")
        print("QUANTUM VQC CV SUMMARY")
        print(f"{'='*60}")
        print(f"  Per-fold F1 : {[f'{f:.4f}' for f in fold_f1s]}")
        print(f"  Mean F1     : {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")
        print(f"  Aggregated F1: {aggregated['macro_f1']:.4f}")

        precision, recall, f1, support = precision_recall_fscore_support(
            all_test_labels, all_test_preds, average=None, zero_division=0
        )
        print(f"\n{'='*60}")
        print("QUANTUM VQC Per-Class Metrics")
        print(f"{'='*60}")
        print(f"\n{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-"*50)
        for i, cls in enumerate(["DOWN", "FLAT", "UP"]):
            print(f"{cls:<10} {precision[i]:>10.4f} {recall[i]:>10.4f} "
                  f"{f1[i]:>10.4f} {int(support[i]):>10}")

        cm = confusion_matrix(all_test_labels, all_test_preds)
        print(f"\nConfusion Matrix:")
        print("Predicted ->  DOWN  FLAT    UP")
        for i, row in enumerate(cm):
            cls = ["DOWN", "FLAT", "UP"][i]
            print(f"  {cls:<6}    {row[0]:>4}  {row[1]:>4}  {row[2]:>4}")

        # Comparison with LSTM 
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        print(f"{'Model':<25} {'Macro-F1':>10}")
        print("-"*35)
        print(f"{'Quantum VQC':<25} {aggregated['macro_f1']:>10.4f}")
        print(f"{'Classical LSTM (ref)':<25} {'~0.3675':>10}")
        print(f"{'Random baseline':<25} {'~0.3333':>10}")

        return {"test_metrics": aggregated, "fold_metrics": fold_metrics}
    

    def save(self) -> None:
        self.bridge.save(self.QUANTUM_MODEL_PATH)
        scaler_path = SCALER_PATH.replace("scaler.pkl", "quantum_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self._best_scaler, f)
        print(f"[QuantumTrainer] Scaler saved to {scaler_path}")

    def load(self) -> None:
        from quantum_bridge import QuantumVQCBridge
        self.bridge = QuantumVQCBridge.load(self.QUANTUM_MODEL_PATH)
        scaler_path = SCALER_PATH.replace("scaler.pkl", "quantum_scaler.pkl")
        with open(scaler_path, "rb") as f:
            self._best_scaler = pickle.load(f)


def main():
    """Main training pipeline."""
    print("="*60)
    print("BTC Price Direction Classifier - Enhanced Training Pipeline")
    print("="*60)
    
    print("\nLoading candle data...")
    df = load_candles_as_dataframe()
    
    if len(df) < 5000:
        print(f"Warning: Only {len(df)} candles available.")
        print("Minimum recommended: 10,000+ candles for meaningful training.")
        
        if len(df) < 2000:
            print("Not enough data to train. Exiting.")
            return None, None, None
    
    print(f"Loaded {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    trainer = Trainer(use_rolling_cv=True)
    
    X, y, timestamps, df_features = trainer.prepare_data(df)
    
    baseline_results = run_baselines(X, y)
    
    results = trainer.train(X, y, timestamps, verbose=True)
    
    print_results_comparison(results['test_metrics'], baseline_results)
    
    lstm_f1 = results['test_metrics']['macro_f1']
    best_baseline_f1 = max(b['macro_f1'] for b in baseline_results)
    
    if lstm_f1 > best_baseline_f1:
        print(f"\n[OK] LSTM outperforms best baseline by {(lstm_f1-best_baseline_f1)*100:.1f}% F1")
    else:
        print(f"\n[!] LSTM does not outperform baselines.")
        print("  Consider: adjusting epsilon, window size, or gathering more data.")
    
    if 'calibration' in results:
        print("\nCalibration Results:")
        print(f"  Temperature: {results['calibration']['temperature']:.4f}")
        print(f"  Min Confidence: {results['calibration']['min_confidence']:.2f}")
        print(f"  Confidence Margin: {results['calibration']['confidence_margin']:.2f}")
    
    print("\nSaving model...")
    trainer.save_model()
    
    return trainer, results, baseline_results


if __name__ == "__main__":
    result = main()
    if result[0] is not None:
        trainer, results, baselines = result
