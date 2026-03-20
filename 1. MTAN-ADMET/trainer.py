import torch
import numpy as np
from metrics_utils import calculate_metrics


class MTANTrainer:
    def __init__(self, model, metadata, base_lr=0.001, k=2):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.is_classification = metadata['is_classification']

        # Skalowanie LR: base_lr * (1 + log(1 + k * x)) [cite: 274, 326]
        self.task_lrs = [base_lr * (1 + np.log(1 + k * x)) for x in metadata['fractions']]

        self.optimizers = [
            torch.optim.AdamW(
                list(model.shared_net.parameters()) + list(model.heads[i].parameters()),
                lr=self.task_lrs[i], weight_decay=1e-4
            ) for i in range(24)
        ]
        self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50) for opt in self.optimizers]

    def _add_grad_noise(self, params, std=0.01):
        """Wstrzykiwanie szumu do gradientów[cite: 297]."""
        for p in params:
            if p.grad is not None:
                p.grad.add_(torch.randn_like(p.grad) * std)

    def train_step(self, loader):
        self.model.train()
        total_loss = 0
        task_order = np.random.permutation(24)  # Perturbative learning [cite: 312]

        for data, targets in loader:
            data, targets = data.to(self.device), targets.to(self.device)
            for t_idx in task_order:
                mask = ~torch.isnan(targets[:, t_idx])
                if mask.sum() == 0: continue

                self.optimizers[t_idx].zero_grad()
                output = self.model(data[mask])[t_idx].squeeze()

                loss_fn = torch.nn.BCEWithLogitsLoss() if self.is_classification[t_idx] else torch.nn.MSELoss()
                loss = loss_fn(output, targets[mask, t_idx])

                loss.backward()
                self._add_grad_noise(self.model.parameters())
                self.optimizers[t_idx].step()
                total_loss += loss.item()
        return total_loss / (len(loader) * 24)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_targets, all_outputs = [], []

        for data, targets in loader:
            data = data.to(self.device)
            outputs = self.model(data)  # Zwraca listę 24 tensorów

            # Konwersja do formatu (batch, 24)
            outputs_tensor = torch.stack([o.squeeze() for o in outputs], dim=1)
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs_tensor.cpu().numpy())

        return calculate_metrics(np.vstack(all_targets), np.vstack(all_outputs), self.is_classification)