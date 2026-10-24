"""Quota-based selection from a fixed pool; not conditional TVAE fitting."""
import numpy as np
import pandas as pd


def binary_quotas(fit_labels, n_rows):
    if type(n_rows) is not int or n_rows < 2:
        raise ValueError('Synthetic size must be an integer of at least two.')
    labels = pd.Series(fit_labels)
    if labels.empty or not labels.isin([0, 1]).all() or set(labels.unique()) != {0, 1}:
        raise ValueError('Fitting labels must contain both binary classes.')
    counts = labels.value_counts().reindex([0, 1]).to_numpy(dtype=int)
    # Integer largest-remainder allocation; class 0 wins an exact tie.
    numerators = counts * n_rows
    quotas = numerators // len(labels)
    remaining = n_rows - int(quotas.sum())
    ordering = sorted(range(2), key=lambda i: (-(int(numerators[i] % len(labels))), i))
    for label in ordering[:remaining]:
        quotas[label] += 1
    if (quotas == 0).any():
        raise ValueError('Requested size rounds a class quota to zero; increase size.')
    return {label: int(quotas[label]) for label in (0, 1)}


def select_pool_rows(pool, quotas, seed):
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError('Seed must be an unsigned integer.')
    if set(quotas) != {0, 1} or any(type(v) is not int or v <= 0 for v in quotas.values()):
        raise ValueError('Both class quotas must be positive integers.')
    if 'target' not in pool or not pool['target'].isin([0, 1]).all():
        raise ValueError('Pool labels must be binary.')
    positions = []
    labels = pool['target'].to_numpy()
    for label in (0, 1):
        available = np.flatnonzero(labels == label)
        if len(available) < quotas[label]:
            raise ValueError(f'Fixed pool lacks enough class {label} rows; do not expand the budget silently.')
        positions.extend(available[:quotas[label]].tolist())
    selected = np.asarray(positions, dtype=int)
    selected = selected[np.random.default_rng(seed).permutation(len(selected))]
    return pool.iloc[selected].reset_index(drop=True).copy(), selected
