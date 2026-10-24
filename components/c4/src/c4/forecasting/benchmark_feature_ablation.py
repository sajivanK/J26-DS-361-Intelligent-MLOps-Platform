"""Training-only feature projections for the frozen PP1 A1-A3 experiments."""
VARIANTS = ('A1', 'A2', 'A3')
QUALITY_COLUMNS = ('c3_fidelity_score','c3_utility_score','c3_target_coverage','c3_prescription_confidence')


def ablation_columns(columns, variant):
    columns = list(columns)
    if len(columns) != len(set(columns)) or not columns:
        raise ValueError('Unique nonempty raw feature names required.')
    if variant == 'A1':
        dropped = [c for c in columns if c.startswith('c1_')]
    elif variant == 'A2':
        dropped = ['c2_diagnosis_stability']
    elif variant == 'A3':
        dropped = [c for c in columns if c in QUALITY_COLUMNS]
        if set(dropped) != set(QUALITY_COLUMNS):
            raise ValueError('All four frozen C3 quality predictors required.')
    else:
        raise ValueError('Only A1, A2 and A3 are feature-ablation variants.')
    if not dropped or not set(dropped) <= set(columns):
        raise ValueError('Required ablation predictors are missing.')
    kept = [c for c in columns if c not in set(dropped)]
    if not kept:
        raise ValueError('Ablation cannot remove all predictors.')
    return kept, dropped


def project_features(features, variant, raw_columns):
    if features.columns.tolist() != list(raw_columns):
        raise ValueError('Raw feature schema/order differs before projection.')
    kept, _ = ablation_columns(raw_columns, variant)
    return features.loc[:, kept].copy()
