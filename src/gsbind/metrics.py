import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(labels, scores, threshold=0.5):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(labels) & np.isfinite(scores)
    labels, scores = labels[valid].astype(int), scores[valid]
    if not len(labels):
        return dict(positions=0, positives=0, aupr=np.nan, auroc=np.nan, mcc=np.nan)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Binary labels required')
    predicted = scores >= threshold
    positive = labels == 1
    true_positive = int(np.count_nonzero(predicted & positive))
    true_negative = int(np.count_nonzero(~predicted & ~positive))
    false_positive = int(np.count_nonzero(predicted & ~positive))
    false_negative = int(np.count_nonzero(~predicted & positive))
    denominator = np.sqrt(float((true_positive + false_positive) * (true_positive + false_negative) *
                                (true_negative + false_positive) * (true_negative + false_negative)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    order = np.argsort(-scores, kind='stable')
    budget = min(10, len(labels))
    hits = int(labels[order[:budget]].sum())
    mixed = len(np.unique(labels)) == 2
    return dict(positions=len(labels), positives=int(labels.sum()), positive_rate=float(labels.mean()),
                aupr=float(average_precision_score(labels, scores)) if labels.sum() else np.nan,
                aupr_mixed=float(average_precision_score(labels, scores)) if mixed else np.nan,
                auroc=float(roc_auc_score(labels, scores)) if mixed else np.nan,
                mcc=(true_positive * true_negative - false_positive * false_negative) / denominator if denominator else 0.,
                mcc_defined=bool(denominator), precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.,
                accuracy=(true_positive + true_negative) / len(labels),
                tp=true_positive, tn=true_negative, fp=false_positive, fn=false_negative,
                threshold=threshold, top10_budget=budget, top10_hits=hits, top10_precision=hits / budget,
                top10_recall=hits / labels.sum() if labels.sum() else np.nan)


def component_difference(frame, baseline, enhanced, component='component', iterations=10000, seed=20261019):
    valid = frame.dropna(subset=[baseline, enhanced])
    differences = valid.assign(difference=valid[enhanced] - valid[baseline]).groupby(component).difference.mean().to_numpy()
    if not len(differences):
        return dict(mean=np.nan, lower=np.nan, upper=np.nan, p=np.nan, components=0)
    generator = np.random.default_rng(seed)
    distribution = differences[generator.integers(0, len(differences), (iterations, len(differences)))].mean(axis=1)
    null = (differences * generator.choice([-1., 1.], (2000, len(differences)))).mean(axis=1)
    lower, upper = np.quantile(distribution, [.025, .975])
    return dict(mean=float(differences.mean()), lower=float(lower), upper=float(upper),
                p=float((1 + np.count_nonzero(abs(null) >= abs(differences.mean()))) / 2001),
                components=len(differences), cases=len(valid), bootstrap=iterations, sign_flips=2000)


def bh_adjust(values):
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    output[order] = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    return output


def group_metrics(frame, groups, label, prediction, threshold=0.5):
    records = []
    for identity, current in frame.groupby(groups, dropna=False, sort=True):
        identity = identity if isinstance(identity, tuple) else (identity,)
        records.append(dict(zip(groups, identity), **binary_metrics(current[label], current[prediction], threshold)))
    return pd.DataFrame(records)
