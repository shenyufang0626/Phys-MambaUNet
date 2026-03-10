import torch
from torch import nn

        
class TD_DQWL(nn.Module): # Time-Decay Dynamic Quantile Weighted Loss
    def __init__(self, omega_t, alpha, beta, c, temporal_beta=0.2):
        super().__init__()
        # keep original parameters
        self.omega_t = omega_t  
        self.alpha = alpha
        self.beta = beta
        self.c = c

        # newly added temporal decay parameter
        self.temporal_beta = temporal_beta  # coefficient controlling the decay speed
        
    def _frame_loss(self, pred, target):
        """core single-frame computation of the original RainfallLoss"""
        w0 = 0.57
        loss_overal1 = torch.sum(((pred >= target)).float() * (1 - w0) * abs(pred - target))
        loss_overal2 = torch.sum(((pred < target)).float() * w0 * abs(pred - target))
        wi = self.alpha * torch.exp(target)
        loss_greater = torch.sum(
            ((pred >= target) & (target >= 0.7)).float() * (1 - self.omega_t) * wi * abs(pred - target))
        loss_less = torch.sum(((pred < target) & (target >= 0.7)).float() * self.omega_t * wi * abs(pred - target))
        num_sample = target.numel()
        frame_loss = (loss_overal1 + loss_overal2) / num_sample + (loss_greater + loss_less) / num_sample
        return frame_loss

    def forward(self, pred_sequence, target_sequence):
        """
        Input:
            pred_sequence: [T, C, H, W] predicted sequence
            target_sequence: [T, C, H, W] ground truth sequence
        """
        total_loss = 0
        T = pred_sequence.shape[0]
        
        # compute temporal decay weights (exponential decay)
        time_weights = torch.exp(-self.temporal_beta * torch.arange(T, device=pred_sequence.device))
        time_weights = time_weights / time_weights.sum()  # normalization
        
        # compute weighted loss frame by frame
        for t in range(T):
            frame_loss = self._frame_loss(pred_sequence[t], target_sequence[t])
            total_loss += frame_loss * time_weights[t]    
        return total_loss