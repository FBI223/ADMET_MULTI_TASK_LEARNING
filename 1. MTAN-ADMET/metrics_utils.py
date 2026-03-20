import numpy as np
from sklearn.metrics import roc_auc_score, r2_score


def calculate_metrics(all_targets, all_outputs, is_classification):
    """Oblicza AUC dla klasyfikacji i R^2 dla regresji dla każdego zadania."""
    stats = []
    for i in range(24):
        targets = all_targets[:, i]
        outputs = all_outputs[:, i]

        # Filtrowanie tylko dostępnych etykiet (non-NaN)
        mask = ~np.isnan(targets)
        t_valid = targets[mask]
        o_valid = outputs[mask]

        if len(t_valid) < 5:  # Zbyt mało danych do statystyki
            stats.append(0.0)
            continue

        try:
            if is_classification[i]:
                # Używamy sigmoid dla wyjść przed AUC
                o_prob = 1 / (1 + np.exp(-o_valid))
                stats.append(roc_auc_score(t_valid, o_prob))
            else:
                stats.append(r2_score(t_valid, o_valid))
        except:
            stats.append(0.0)
    return stats