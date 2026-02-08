import torch
from torch import nn, optim
import torch.nn.functional as F


class TemperatureScaling(nn.Module):
    """Temperature scaling with ranking preservation via MRL loss"""
    
    def __init__(self, init_temp=1.0):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)
    
    def forward(self, logits):
        """Scale logits by temperature"""
        return logits / self.temperature
    
    def set_temperature(self, pos_logits, neg_logits, lr=0.01, max_iters=200, lambda_mrl=1.0):
        """
        Tune temperature using both calibration (BCE) and ranking (MRL) objectives.
        
        Args:
            pos_logits: Raw scores for positive triples [N]
            neg_logits: Raw scores for negative triples [N]
        """
        self.cuda()
        bce_criterion = nn.BCELoss().cuda()
        mrl_criterion = nn.MarginRankingLoss(margin=0.1).cuda()
        
        # Convert to tensors
        if isinstance(pos_logits, torch.Tensor):
            pos_logits = pos_logits.detach().clone().cuda()
        else:
            pos_logits = torch.tensor(pos_logits, dtype=torch.float).cuda()
        
        if isinstance(neg_logits, torch.Tensor):
            neg_logits = neg_logits.detach().clone().cuda()
        else:
            neg_logits = torch.tensor(neg_logits, dtype=torch.float).cuda()
        
        # Store training mode
        training_mode = self.training
        self.eval()
        
        # Initial statistics
        with torch.no_grad():
            pos_probs = torch.sigmoid(self(pos_logits))
            neg_probs = torch.sigmoid(self(neg_logits))
            
            bce_pos = bce_criterion(pos_probs, torch.ones_like(pos_probs))
            bce_neg = bce_criterion(neg_probs, torch.zeros_like(neg_probs))
            before_bce = bce_pos + bce_neg
            
            target = torch.ones(pos_probs.size(0)).cuda()
            before_mrl = mrl_criterion(pos_probs, neg_probs, target)
        
        print(f'Before Temperature Scaling:')
        print(f'  BCE Loss: {before_bce.item():.4f}')
        print(f'  MRL Loss: {before_mrl.item():.4f}')
        print(f'  Initial Temperature: {self.temperature.item():.3f}')
        
        # Optimizer
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iters)
        
        # Tracking
        n_iter = [0]
        best_loss = [float('inf')]
        patience_counter = [0]
        patience = 10
        should_stop = [False]
        
        def eval_closure():
            if should_stop[0]:
                return best_loss[0]
            
            optimizer.zero_grad()
            
            # Apply temperature scaling
            pos_scaled = self(pos_logits)
            neg_scaled = self(neg_logits)
            
            pos_probs = torch.sigmoid(pos_scaled)
            neg_probs = torch.sigmoid(neg_scaled)
            
            # BCE Loss (calibration)
            bce_pos = bce_criterion(pos_probs, torch.ones_like(pos_probs))
            bce_neg = bce_criterion(neg_probs, torch.zeros_like(neg_probs))
            bce_loss = bce_pos + bce_neg
            
            # MRL Loss (ranking preservation)
            target = torch.ones(pos_probs.size(0)).cuda()
            mrl_loss = mrl_criterion(pos_probs, neg_probs, target)
            
            # Combined loss (prevents negative temperature!)
            loss = bce_loss + mrl_loss * lambda_mrl
            loss.backward()
            
            n_iter[0] += 1
            current_loss = loss.detach().item()
            
            # Early stopping
            if current_loss < best_loss[0] - 1e-6:
                best_loss[0] = current_loss
                patience_counter[0] = 0
            else:
                patience_counter[0] += 1
                if patience_counter[0] >= patience:
                    print(f"Early stopping at iteration {n_iter[0]}")
                    should_stop[0] = True
            
            return loss
        
        # Optimize
        optimizer.step(eval_closure)
        
        # Final statistics
        with torch.no_grad():
            pos_probs = torch.sigmoid(self(pos_logits))
            neg_probs = torch.sigmoid(self(neg_logits))
            
            bce_pos = bce_criterion(pos_probs, torch.ones_like(pos_probs))
            bce_neg = bce_criterion(neg_probs, torch.zeros_like(neg_probs))
            after_bce = bce_pos + bce_neg
            
            target = torch.ones(pos_probs.size(0)).cuda()
            after_mrl = mrl_criterion(pos_probs, neg_probs, target)
        
        print(f'\nAfter Temperature Scaling:')
        print(f'  BCE Loss: {after_bce.item():.4f} (Δ {after_bce.item() - before_bce.item():+.4f})')
        print(f'  MRL Loss: {after_mrl.item():.4f} (Δ {after_mrl.item() - before_mrl.item():+.4f})')
        print(f'  Optimal Temperature: {self.temperature.item():.3f}')
        print(f'  Optimization steps: {n_iter[0]}')
        
        # Check for issues
        if self.temperature.item() < 0:
            print(f'\nWARNING: Temperature is negative! Rankings will be inverted!')
            print(f'   MRL loss should have prevented this. Check your implementation.')
        
        # Ranking violations
        with torch.no_grad():
            violations = (pos_probs <= neg_probs).sum().item()
            total = pos_probs.size(0)
            print(f'  Ranking violations: {violations}/{total} ({100*violations/total:.1f}%)')
        
        # Restore mode
        self.train(training_mode)
        return self
