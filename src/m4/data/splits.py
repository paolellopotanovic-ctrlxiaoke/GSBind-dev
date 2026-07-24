"""Deterministic grouped data splitting and class-balance statistics."""

import torch


def estimate_pos_weight(rows, factor=1.0):
    positive = 0
    valid = 0
    for row in rows:
        labels = [int(value) for value in row.get("labels", "").split(";") if value != ""]
        masks = [int(value) for value in row.get("label_mask", "").split(";") if value != ""]
        for label, mask in zip(labels, masks):
            if mask:
                valid += 1
                positive += int(label)
    rate = min(max(positive / max(valid, 1), 1e-4), 1.0)
    return factor * (1.0 - rate) / rate, {
        "valid": valid,
        "positive": positive,
        "positive_rate": rate,
    }


def split_rows(rows, val_fraction=0.1, seed=42):
    systems = sorted({row["system_id"] for row in rows})
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(len(systems), generator=generator).tolist()
    shuffled = [systems[index] for index in permutation]
    validation_count = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    validation_systems = set(shuffled[:validation_count])
    train = [row for row in rows if row["system_id"] not in validation_systems]
    validation = [row for row in rows if row["system_id"] in validation_systems]
    return train, validation


def split_rows_grouped(rows, val_fraction=0.1, seed=42, group_key="system_id"):
    if group_key == "system_id":
        return split_rows(rows, val_fraction=val_fraction, seed=seed)

    def key_for(row):
        if group_key == "cluster_id":
            return row.get("cluster_id") or row.get("system_id")
        if group_key == "cluster_id_RL":
            return "|".join(
                [row.get("cluster_id", ""), row.get("cluster_id_R", ""), row.get("cluster_id_L", "")]
            ) or row.get("system_id")
        if group_key == "receptor_uniprot":
            side = row.get("receptor_side", "")
            return row.get("receptor_uniprot") or row.get(f"uniprot_{side}") or row.get("system_id")
        if group_key == "partner_uniprot":
            side = row.get("partner_side", "")
            return row.get("partner_uniprot") or row.get(f"uniprot_{side}") or row.get("system_id")
        raise ValueError(f"Unsupported val split group key: {group_key}")

    groups = sorted({key_for(row) for row in rows})
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(len(groups), generator=generator).tolist()
    shuffled = [groups[index] for index in permutation]
    validation_count = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    validation_groups = set(shuffled[:validation_count])
    train = [row for row in rows if key_for(row) not in validation_groups]
    validation = [row for row in rows if key_for(row) in validation_groups]
    return train, validation
