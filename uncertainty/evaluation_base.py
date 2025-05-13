import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    def __init__(self, model, device=None, flatten = True):
        self.model = model.to(device or next(model.parameters()).device)
        self.device = device or next(model.parameters()).device
        self.flatten = flatten

    @abstractmethod
    def compute_score(self, x):
        """
        Compute uncertainty score for a single sample x.
        Must be implemented by subclasses.
        """
        pass

    def compute_batch(self, X):
        scores = []
        for x in X:
            if self.flatten:
                x = x.view(-1)
            # normalize to reduce scale bias
            x = (x - x.mean()) / (x.std() + 1e-8)
            scores.append(self.compute_score(x))
        return np.array(scores)

    def compute_auroc(self, X_id, X_ood, plot = False):
        id_scores = self.compute_batch(X_id)
        ood_scores = self.compute_batch(X_ood)
        if plot:
            self.plot_histogram(id_scores, ood_scores)
        return self._compute_auroc(id_scores, ood_scores)

    def _compute_auroc(self, id_scores, ood_scores):
        y_true = np.concatenate([
            np.zeros(len(id_scores)),
            np.ones(len(ood_scores))
        ])
        y_scores = np.concatenate([
            id_scores,
            ood_scores
        ])
        return roc_auc_score(y_true, y_scores)

    def plot_histogram(self, id_scores, ood_scores):
        plt.hist(id_scores, bins=30, alpha=0.6, label='ID')
        plt.hist(ood_scores, bins=30, alpha=0.6, label='OoD')
        plt.legend()
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Uncertainty Score Distribution')
        plt.show()
