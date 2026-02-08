from sklearn.isotonic import IsotonicRegression
import numpy as np
import torch

class IsotonicCalibrator:
    
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')

    def fit(self, logits, labels):
        self.calibrator.fit(logits, labels)

    def forward(self, logits):
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        calibrated = self.calibrator.predict(logits)
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return torch.from_numpy(calibrated).float()
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)