import torch
from torch import nn, optim
import torch.nn.functional as F

class PlattScaling(nn.Module):
    """
    Platt Scaling: sigmoid(A*z + B) for calibration with ranking preservation.
    This is the method used in the thesis (2 learnable parameters).
    """
    
    def __init__(self, init_A=1.0, init_B=0.0):
        super(PlattScaling, self).__init__()
        self.A = nn.Parameter(torch.ones(1) * init_A)
        self.B = nn.Parameter(torch.zeros(1) * init_B)
    
    def forward(self, logits):
        """Apply Platt scaling: A*z + B"""
        return self.A * logits + self.B
    
    def set_parameters(self, pos_logits, neg_logits, lr=0.01, max_iters=200, lambda_mrl=1.0):
        """
        Tune A and B parameters using both calibration (BCE) and ranking (MRL) objectives.
        Implements the combined loss from thesis Equation 5.1.
        
        Args:
            pos_logits: Raw scores for positive triples [N]
            neg_logits: Raw scores for negative triples [N]
            lr: Learning rate for LBFGS optimizer
            max_iters: Maximum optimization iterations
        """
        self.cuda()
        bce_criterion = nn.BCELoss().cuda()
        mrl_criterion = nn.MarginRankingLoss(margin=0.1).cuda()
        
        # Convert to tensors and move to GPU
        if isinstance(pos_logits, torch.Tensor):
            pos_logits = pos_logits.detach().clone().cuda()
        else:
            pos_logits = torch.tensor(pos_logits, dtype=torch.float).cuda()
        
        if isinstance(neg_logits, torch.Tensor):
            neg_logits = neg_logits.detach().clone().cuda()
        else:
            neg_logits = torch.tensor(neg_logits, dtype=torch.float).cuda()
        
        # Store original training mode
        training_mode = self.training
        self.eval()
        
        # Calculate initial loss (before calibration)
        with torch.no_grad():
            pos_probs = torch.sigmoid(self(pos_logits))
            neg_probs = torch.sigmoid(self(neg_logits))
            
            bce_pos = bce_criterion(pos_probs, torch.ones_like(pos_probs))
            bce_neg = bce_criterion(neg_probs, torch.zeros_like(neg_probs))
            before_bce = bce_pos + bce_neg
            
            target = torch.ones(pos_probs.size(0)).cuda()
            before_mrl = mrl_criterion(pos_probs, neg_probs, target)
            before_total = before_bce + lambda_mrl * before_mrl
        
        print(f'Before Platt Scaling:')
        print(f'  BCE Loss: {before_bce.item():.4f}')
        print(f'  MRL Loss: {before_mrl.item():.4f}')
        print(f'  Total Loss: {before_total.item():.4f}')
        
        optimizer = optim.LBFGS([self.A, self.B], lr=lr, max_iter=max_iters)
        

        n_iter = [0]
        best_loss = [float('inf')]
        patience_counter = [0]
        patience = 10
        should_stop = [False]
        
        def eval_closure():
            """Closure for LBFGS optimizer"""
            if should_stop[0]:
                return best_loss[0]
            
            optimizer.zero_grad()
            
            pos_scaled = self(pos_logits)
            neg_scaled = self(neg_logits)
            
            pos_probs = torch.sigmoid(pos_scaled)
            neg_probs = torch.sigmoid(neg_scaled)
            
            bce_pos = bce_criterion(pos_probs, torch.ones_like(pos_probs))
            bce_neg = bce_criterion(neg_probs, torch.zeros_like(neg_probs))
            bce_loss = bce_pos + bce_neg
            
            target = torch.ones(pos_probs.size(0)).cuda()
            mrl_loss = mrl_criterion(pos_probs, neg_probs, target)
            
            # Combined loss (thesis Equation 5.1)
            loss = bce_loss + mrl_loss
            loss.backward()
            
            n_iter[0] += 1
            current_loss = loss.detach().item()
            
            # Early stopping logic
            if current_loss < best_loss[0] - 1e-6:
                best_loss[0] = current_loss
                patience_counter[0] = 0
            else:
                patience_counter[0] += 1
                if patience_counter[0] >= patience:
                    print(f"Early stopping at iteration {n_iter[0]}")
                    should_stop[0] = True
            
            return loss
        
        # Run optimization
        optimizer.step(eval_closure)
        
        # Calculate final loss (after calibration)
        with torch.no_grad():
            pos_probs = torch.sigmoid(self(pos_logits))
            neg_probs = torch.sigmoid(self(neg_logits))
            
            bce_pos = bce_criterion(pos_probs, torch.ones_like(pos_probs))
            bce_neg = bce_criterion(neg_probs, torch.zeros_like(neg_probs))
            after_bce = bce_pos + bce_neg
            
            target = torch.ones(pos_probs.size(0)).cuda()
            after_mrl = mrl_criterion(pos_probs, neg_probs, target)
            after_total = after_bce + after_mrl
        
        print(f'\nAfter Platt Scaling:')
        print(f'  BCE Loss: {after_bce.item():.4f} (Δ {after_bce.item() - before_bce.item():+.4f})')
        print(f'  MRL Loss: {after_mrl.item():.4f} (Δ {after_mrl.item() - before_mrl.item():+.4f})')
        print(f'  Total Loss: {after_total.item():.4f} (Δ {after_total.item() - before_total.item():+.4f})')
        print(f'  Step count: {n_iter[0]}')

    
        # Verify ranking preservation
        with torch.no_grad():
            violations = (pos_probs <= neg_probs).sum().item()
            total = pos_probs.size(0)
            print(f'  Ranking violations: {violations}/{total} ({100*violations/total:.1f}%)')
            
            # Additional diagnostics
            print(f'\nCalibrated probability statistics:')
            print(f'  Positives - Mean: {pos_probs.mean().item():.3f}, Std: {pos_probs.std().item():.3f}')
            print(f'  Negatives - Mean: {neg_probs.mean().item():.3f}, Std: {neg_probs.std().item():.3f}')
        
        if self.A.item() < 0:
            print(f'\n WARNING: A is negative ({self.A.item():.4f})! Rankings may be inverted.')

        self.train(training_mode)
        return self

