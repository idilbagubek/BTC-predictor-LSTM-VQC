import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from config import model_config


class LSTMClassifier(nn.Module):
    """
    Basic LSTM-based classifier for 3-class price direction prediction.
    """
    
    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = None,
        num_layers: int = None,
        dropout: float = None,
        num_classes: int = None
    ):
        super(LSTMClassifier, self).__init__()
        
        input_size = input_size or model_config.input_size
        hidden_size = hidden_size or model_config.hidden_size
        num_layers = num_layers or model_config.num_layers
        dropout = dropout or model_config.dropout
        num_classes = num_classes or model_config.num_classes
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (not probabilities) for stable focal loss."""
        lstm_out, (h_n, c_n) = self.lstm(x)
        final_hidden = h_n[-1]
        out = self.layer_norm(final_hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits
    
    def predict_proba(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Get class probabilities with optional temperature scaling."""
        logits = self.forward(x)
        return F.softmax(logits / temperature, dim=-1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)


class ScaledBilinearAttention(nn.Module):
    """
    Scaled bilinear attention: e_t = (h_t^T W_a h_final) / sqrt(H)
    """
    
    def __init__(self, hidden_size: int):
        super(ScaledBilinearAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = math.sqrt(hidden_size)
        
    def forward(
        self, 
        lstm_outputs: torch.Tensor, 
        final_hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        projected = self.W_a(lstm_outputs)
        scores = torch.bmm(projected, final_hidden.unsqueeze(-1)).squeeze(-1)
        scores = scores / self.scale
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_outputs).squeeze(1)
        
        return context, attn_weights


class QKScaledDotAttention(nn.Module):
    """
    Query-Key scaled dot-product attention:
    q = W_q h_final, k_t = W_k h_t
    e_t = (q^T k_t) / sqrt(H)
    """
    
    def __init__(self, hidden_size: int):
        super(QKScaledDotAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = math.sqrt(hidden_size)
        
    def forward(
        self, 
        lstm_outputs: torch.Tensor, 
        final_hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.W_q(final_hidden)
        keys = self.W_k(lstm_outputs)
        scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)
        scores = scores / self.scale
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_outputs).squeeze(1)
        
        return context, attn_weights


class AttentionLSTMClassifier(nn.Module):
    """
    LSTM with attention pooling for better temporal credit assignment.
    
    Instead of relying solely on the final hidden state, attention learns
    which timesteps in the sequence are most informative for prediction.
    
    Supports two attention variants:
    - scaled_bilinear
    - qk_scaled_dot
    
    Final classifier input is concat(context, h_final) for richer representation.
    """
    
    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = None,
        num_layers: int = None,
        dropout: float = None,
        num_classes: int = None,
        attention_variant: str = None
    ):
        super(AttentionLSTMClassifier, self).__init__()
        
        input_size = input_size or model_config.input_size
        hidden_size = hidden_size or model_config.hidden_size
        num_layers = num_layers or model_config.num_layers
        dropout = dropout or model_config.dropout
        num_classes = num_classes or model_config.num_classes
        attention_variant = attention_variant or model_config.attention_variant
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_variant = attention_variant
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        if attention_variant == "scaled_bilinear":
            self.attention = ScaledBilinearAttention(hidden_size)
        elif attention_variant == "qk_scaled_dot":
            self.attention = QKScaledDotAttention(hidden_size)
        else:
            raise ValueError(f"Unknown attention variant: {attention_variant}")
        
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass returning logits (for focal loss + calibration).
        
        Args:
            x: [batch, seq_len, input_size]
            return_attention: If True, also return attention weights
        Returns:
            logits: [batch, num_classes]
            attn_weights: [batch, seq_len] (optional)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        final_hidden = h_n[-1]
        context, attn_weights = self.attention(lstm_out, final_hidden)
        combined = torch.cat([context, final_hidden], dim=-1)
        out = self.layer_norm(combined)
        out = self.dropout(out)
        logits = self.fc(out)
        
        if return_attention:
            return logits, attn_weights
        return logits
    
    def predict_proba(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Get class probabilities with optional temperature scaling."""
        logits = self.forward(x)
        return F.softmax(logits / temperature, dim=-1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability."""
        _, attn_weights = self.forward(x, return_attention=True)
        return attn_weights


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Computed from logits for numerical stability using log_softmax.
    """
    
    def __init__(
        self, 
        gamma: float = 2.0, 
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, num_classes]
            targets: [batch] (class indices)
        Returns:
            loss: scalar
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        pt = (probs * targets_one_hot).sum(dim=-1)
        log_pt = (log_probs * targets_one_hot).sum(dim=-1)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = -alpha_t * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> Dict:
    return {
        'type': model.__class__.__name__,
        'trainable_params': count_parameters(model),
        'layers': str(model)
    }


def create_model(use_attention: bool = None, attention_variant: str = None) -> nn.Module:
    """Factory function to create the appropriate model."""
    use_attention = use_attention if use_attention is not None else model_config.use_attention
    attention_variant = attention_variant or model_config.attention_variant
    
    if use_attention and attention_variant != "none":
        return AttentionLSTMClassifier(attention_variant=attention_variant)
    else:
        return LSTMClassifier()


if __name__ == "__main__":
    print("Testing LSTM Classifier...")
    
    model = LSTMClassifier()
    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    batch_size = 32
    seq_len = 720
    input_size = model_config.input_size
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    logits = model(x)
    probs = model.predict_proba(x)
    preds = model.predict(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape}")
    print(f"Preds shape: {preds.shape}")
    
    print("\n" + "="*50)
    print("Testing Attention LSTM Classifier (scaled_bilinear)...")
    
    attn_model_bilinear = AttentionLSTMClassifier(attention_variant="scaled_bilinear")
    print(f"Trainable parameters: {count_parameters(attn_model_bilinear):,}")
    
    logits, attn_weights = attn_model_bilinear(x, return_attention=True)
    print(f"Output shape: {logits.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention sum (should be ~1.0): {attn_weights[0].sum().item():.4f}")
    
    print("\n" + "="*50)
    print("Testing Attention LSTM Classifier (qk_scaled_dot)...")
    
    attn_model_qk = AttentionLSTMClassifier(attention_variant="qk_scaled_dot")
    print(f"Trainable parameters: {count_parameters(attn_model_qk):,}")
    
    logits, attn_weights = attn_model_qk(x, return_attention=True)
    print(f"Output shape: {logits.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    print("\n" + "="*50)
    print("Testing Focal Loss...")
    
    targets = torch.randint(0, 3, (batch_size,))
    
    ce_loss = F.cross_entropy(logits, targets)
    print(f"CrossEntropy loss: {ce_loss.item():.4f}")
    
    focal = FocalLoss(gamma=2.0)
    fl_loss = focal(logits, targets)
    print(f"Focal loss (gamma=2.0): {fl_loss.item():.4f}")
    
    alpha = torch.tensor([1.0, 0.5, 1.0])
    focal_alpha = FocalLoss(gamma=2.0, alpha=alpha)
    fl_alpha_loss = focal_alpha(logits, targets)
    print(f"Focal loss (gamma=2.0, alpha): {fl_alpha_loss.item():.4f}")
    
    print("\n" + "="*50)
    print("Testing Temperature Scaling...")
    
    probs_t1 = attn_model_qk.predict_proba(x, temperature=1.0)
    probs_t2 = attn_model_qk.predict_proba(x, temperature=2.0)
    probs_t05 = attn_model_qk.predict_proba(x, temperature=0.5)
    
    print(f"T=1.0 max prob: {probs_t1[0].max().item():.4f}")
    print(f"T=2.0 max prob: {probs_t2[0].max().item():.4f} (smoother)")
    print(f"T=0.5 max prob: {probs_t05[0].max().item():.4f} (sharper)")
    
    print("\nAll tests passed!")
