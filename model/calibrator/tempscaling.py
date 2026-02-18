import torch
from torch import nn, optim
import torch.nn.functional as F


def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemperatureScaling(nn.Module):
    """Temperature scaling with ranking preservation via MRL loss"""

    def __init__(self, init_temp=1.0, device=None):
        super(TemperatureScaling, self).__init__()
        self.device = device if device is not None else get_default_device()
        self.temperature = nn.Parameter(torch.ones(1, device=self.device) * init_temp)

    def forward(self, logits):
        """Scale logits by temperature"""
        return logits / self.temperature

    def set_temperature(self, pos_logits, neg_logits, lr=0.01, max_iters=200, lambda_mrl=1.0):
        """
        Tune temperature using both calibration (BCE) and ranking (MRL) objectives.
        """
        device = self.device
        self.to(device)

        bce_criterion = nn.BCELoss().to(device)
        mrl_criterion = nn.MarginRankingLoss(margin=0.1).to(device)

        if pos_logits.size(0) != neg_logits.size(0):
            pos_logits = pos_logits.repeat((neg_logits.size(0) // pos_logits.size(0), 1))

        # Convert to tensors and move to device
        if isinstance(pos_logits, torch.Tensor):
            pos_logits = pos_logits.detach().clone().to(device)
        else:
            pos_logits = torch.tensor(pos_logits, dtype=torch.float, device=device)

        if isinstance(neg_logits, torch.Tensor):
            neg_logits = neg_logits.detach().clone().to(device)
        else:
            neg_logits = torch.tensor(neg_logits, dtype=torch.float, device=device)

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

            target = torch.ones(pos_probs.size(0), device=device)
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
        print(f" Lambda value: {lambda_mrl}")

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
            target = torch.ones(pos_probs.size(0), device=device)
            mrl_loss = mrl_criterion(pos_probs, neg_probs, target)

            # Combined loss
            loss = bce_loss + lambda_mrl * mrl_loss
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

            target = torch.ones(pos_probs.size(0), device=device)
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


class TemperatureMCDropout(nn.Module):
    """Temperature scaling with MC Dropout support."""

    def __init__(self, init_temp=1.0, device=None):
        super(TemperatureMCDropout, self).__init__()
        self.device = device if device is not None else get_default_device()
        self.temperature = nn.Parameter(torch.ones(1, device=self.device) * init_temp)

    def forward(self, logits_stack):
        """ logits_stack: N x MC_SAMPLES """
        logits_stack = logits_stack.to(self.device)
        probs = torch.sigmoid(logits_stack / self.temperature)
        return probs.mean(dim=1)

    def set_temperature(self, pos_logits, neg_logits, lr=0.01, max_iters=200, lambda_mrl=1.0):
        device = self.device
        self.to(device)

        bce_criterion = nn.BCELoss().to(device)
        mrl_criterion = nn.MarginRankingLoss(margin=0.1).to(device)

        if pos_logits.size(0) != neg_logits.size(0):
            pos_logits = pos_logits.repeat((neg_logits.size(0) // pos_logits.size(0), 1))

        if isinstance(pos_logits, torch.Tensor):
            pos_logits = pos_logits.detach().clone().to(device)
        else:
            pos_logits = torch.tensor(pos_logits, dtype=torch.float, device=device)

        if isinstance(neg_logits, torch.Tensor):
            neg_logits = neg_logits.detach().clone().to(device)
        else:
            neg_logits = torch.tensor(neg_logits, dtype=torch.float, device=device)

        training_mode = self.training
        self.eval()

        with torch.no_grad():
            pos_probs = self(pos_logits)
            neg_probs = self(neg_logits)

            bce_pos = bce_criterion(pos_probs, torch.ones_like(pos_probs))
            bce_neg = bce_criterion(neg_probs, torch.zeros_like(neg_probs))
            before_bce = bce_pos + bce_neg

            target = torch.ones(pos_probs.size(0), device=device)
            before_mrl = mrl_criterion(pos_probs, neg_probs, target)
            before_total = before_bce + lambda_mrl * before_mrl

        print(f'Before Temperature Scaling (MC Dropout):')
        print(f'  BCE Loss: {before_bce.item():.4f}')
        print(f'  MRL Loss: {before_mrl.item():.4f}')
        print(f'  Total Loss: {before_total.item():.4f}')

        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iters)

        n_iter = [0]
        best_loss = [float('inf')]
        patience_counter = [0]
        patience = 10
        should_stop = [False]

        def eval_closure():
            if should_stop[0]:
                return best_loss[0]

            optimizer.zero_grad()

            pos_probs = self(pos_logits)
            neg_probs = self(neg_logits)

            bce_pos = bce_criterion(pos_probs, torch.ones_like(pos_probs))
            bce_neg = bce_criterion(neg_probs, torch.zeros_like(neg_probs))
            bce_loss = bce_pos + bce_neg

            target = torch.ones(pos_probs.size(0), device=device)
            mrl_loss = mrl_criterion(pos_probs, neg_probs, target)

            loss = bce_loss + lambda_mrl * mrl_loss
            loss.backward()

            n_iter[0] += 1
            current_loss = loss.detach().item()

            if current_loss < best_loss[0] - 1e-6:
                best_loss[0] = current_loss
                patience_counter[0] = 0
            else:
                patience_counter[0] += 1
                if patience_counter[0] >= patience:
                    print(f"Early stopping at iteration {n_iter[0]}")
                    should_stop[0] = True

            return loss

        optimizer.step(eval_closure)

        with torch.no_grad():
            pos_probs = self(pos_logits)
            neg_probs = self(neg_logits)

            bce_pos = bce_criterion(pos_probs, torch.ones_like(pos_probs))
            bce_neg = bce_criterion(neg_probs, torch.zeros_like(neg_probs))
            after_bce = bce_pos + bce_neg

            target = torch.ones(pos_probs.size(0), device=device)
            after_mrl = mrl_criterion(pos_probs, neg_probs, target)
            after_total = after_bce + lambda_mrl * after_mrl

        print(f'\nAfter Temperature Scaling (MC Dropout):')
        print(f'  BCE Loss: {after_bce.item():.4f} (Δ {after_bce.item() - before_bce.item():+.4f})')
        print(f'  MRL Loss: {after_mrl.item():.4f} (Δ {after_mrl.item() - before_mrl.item():+.4f})')
        print(f'  Total Loss: {after_total.item():.4f} (Δ {after_total.item() - before_total.item():+.4f})')
        print(f'  Step count: {n_iter[0]}')
        print(f'  Optimal Temperature: {self.temperature.item():.3f}')

        with torch.no_grad():
            violations = (pos_probs <= neg_probs).sum().item()
            total = pos_probs.size(0)
            print(f'  Ranking violations: {violations}/{total} ({100*violations/total:.1f}%)')

            print(f'\nCalibrated probability statistics:')
            print(f'  Positives - Mean: {pos_probs.mean().item():.3f}, Std: {pos_probs.std().item():.3f}')
            print(f'  Negatives - Mean: {neg_probs.mean().item():.3f}, Std: {neg_probs.std().item():.3f}')

        if self.temperature.item() < 0:
            print(f'\n WARNING: Temperature is negative ({self.temperature.item():.4f})! Rankings may be inverted.')

        self.train(training_mode)
        return self


class TemperatureEnsemble(nn.Module):
    """Temperature scaling for ensembles."""

    def __init__(self, init_temp=1.0, device=None):
        super(TemperatureEnsemble, self).__init__()
        self.device = device if device is not None else get_default_device()
        self.temperature = nn.Parameter(torch.ones(1, device=self.device) * init_temp)

    def forward(self, logits_stack):
        """ logits_stack: N x MC_SAMPLES """
        logits_stack = logits_stack.to(self.device)
        probs = torch.sigmoid(logits_stack / self.temperature)
        return probs

    def set_temperature(self, pos_logits, neg_logits, lr=0.01, max_iters=200, lambda_mrl=1.0):
        device = self.device
        self.to(device)

        bce_criterion = nn.BCELoss().to(device)
        mrl_criterion = nn.MarginRankingLoss(margin=0.1).to(device)

        if pos_logits.size(0) != neg_logits.size(0):
            pos_logits = pos_logits.repeat((neg_logits.size(0) // pos_logits.size(0), 1))

        if isinstance(pos_logits, torch.Tensor):
            pos_logits = pos_logits.detach().clone().to(device)
        else:
            pos_logits = torch.tensor(pos_logits, dtype=torch.float, device=device)

        if isinstance(neg_logits, torch.Tensor):
            neg_logits = neg_logits.detach().clone().to(device)
        else:
            neg_logits = torch.tensor(neg_logits, dtype=torch.float, device=device)

        training_mode = self.training
        self.eval()

        with torch.no_grad():
            pos_probs = self(pos_logits)
            neg_probs = self(neg_logits)

            bce_pos = bce_criterion(pos_probs, torch.ones_like(pos_probs))
            bce_neg = bce_criterion(neg_probs, torch.zeros_like(neg_probs))
            before_bce = bce_pos + bce_neg

            target = torch.ones(pos_probs.size(0), device=device)
            before_mrl = mrl_criterion(pos_probs, neg_probs, target)
            before_total = before_bce + lambda_mrl * before_mrl

        print(f'Before Temperature Scaling (Ensemble):')
        print(f'  BCE Loss: {before_bce.item():.4f}')
        print(f'  MRL Loss: {before_mrl.item():.4f}')
        print(f'  Total Loss: {before_total.item():.4f}')

        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iters)

        n_iter = [0]
        best_loss = [float('inf')]
        patience_counter = [0]
        patience = 10
        should_stop = [False]

        def eval_closure():
            if should_stop[0]:
                return best_loss[0]

            optimizer.zero_grad()

            pos_probs = self(pos_logits)
            neg_probs = self(neg_logits)

            bce_pos = bce_criterion(pos_probs, torch.ones_like(pos_probs))
            bce_neg = bce_criterion(neg_probs, torch.zeros_like(neg_probs))
            bce_loss = bce_pos + bce_neg

            target = torch.ones(pos_probs.size(0), device=device)
            mrl_loss = mrl_criterion(pos_probs, neg_probs, target)

            loss = bce_loss + lambda_mrl * mrl_loss
            loss.backward()

            n_iter[0] += 1
            current_loss = loss.detach().item()

            if current_loss < best_loss[0] - 1e-6:
                best_loss[0] = current_loss
                patience_counter[0] = 0
            else:
                patience_counter[0] += 1
                if patience_counter[0] >= patience:
                    print(f"Early stopping at iteration {n_iter[0]}")
                    should_stop[0] = True

            return loss

        optimizer.step(eval_closure)

        with torch.no_grad():
            pos_probs = self(pos_logits)
            neg_probs = self(neg_logits)

            bce_pos = bce_criterion(pos_probs, torch.ones_like(pos_probs))
            bce_neg = bce_criterion(neg_probs, torch.zeros_like(neg_probs))
            after_bce = bce_pos + bce_neg

            target = torch.ones(pos_probs.size(0), device=device)
            after_mrl = mrl_criterion(pos_probs, neg_probs, target)
            after_total = after_bce + lambda_mrl * after_mrl

        print(f'\nAfter Temperature Scaling (Ensemble):')
        print(f'  BCE Loss: {after_bce.item():.4f} (Δ {after_bce.item() - before_bce.item():+.4f})')
        print(f'  MRL Loss: {after_mrl.item():.4f} (Δ {after_mrl.item() - before_mrl.item():+.4f})')
        print(f'  Total Loss: {after_total.item():.4f} (Δ {after_total.item() - before_total.item():+.4f})')
        print(f'  Step count: {n_iter[0]}')
        print(f'  Optimal Temperature: {self.temperature.item():.3f}')

        with torch.no_grad():
            violations = (pos_probs <= neg_probs).sum().item()
            total = pos_probs.size(0)
            print(f'  Ranking violations: {violations}/{total} ({100*violations/total:.1f}%)')

            print(f'\nCalibrated probability statistics:')
            print(f'  Positives - Mean: {pos_probs.mean().item():.3f}, Std: {pos_probs.std().item():.3f}')
            print(f'  Negatives - Mean: {neg_probs.mean().item():.3f}, Std: {neg_probs.std().item():.3f}')

        if self.temperature.item() < 0:
            print(f'\n WARNING: Temperature is negative ({self.temperature.item():.4f})! Rankings may be inverted.')

        self.train(training_mode)
        return self
