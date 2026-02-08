from typing import Literal
from sklearn.isotonic import IsotonicRegression
import numpy as np
import torch

class IsotonicCalibrator:
    
    def __init__(self, out_of_bounds: Literal['clip', 'nan', 'raise'] = 'clip'):
        self.calibrator = IsotonicRegression(out_of_bounds=out_of_bounds)

    def fit(self, logits, labels):
        self.calibrator.fit(logits, labels)

    def forward(self, logits):
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        calibrated = self.calibrator.predict(logits)
        return torch.from_numpy(calibrated).float()
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def predict(self, logits):
        return self.calibrator.predict(logits)