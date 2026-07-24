"""Auxiliary M4 objectives preserved from the validated training recipe."""

import torch

try:
    from scipy.optimize import linear_sum_assignment
except ModuleNotFoundError:
    linear_sum_assignment = None

from ..data.soft_labels import masked_soft_bce, surface_soft_labels
from .runtime import masked_bce


def pocket_mass_loss_from_batch(output, batch, pos_weight, args):
    if (
        str(getattr(args, "stochastic_head_mode", "none")) != "poisson_pocket"
        or float(getattr(args, "pocket_mass_loss_weight", 0.0)) <= 0
    ):
        return output["logits"].sum() * 0.0
    embeddings = output.get("residue_embeddings")
    if embeddings is None:
        return output["logits"].sum() * 0.0
    logits = output["logits"]
    labels = batch["y"].to(device=logits.device, dtype=logits.dtype)
    mask = batch["mask"].to(device=logits.device, dtype=logits.dtype)
    counts = batch["residue_counts"].detach().cpu().tolist()
    losses = []
    offset = 0
    topk = max(1, int(getattr(args, "pocket_mass_topk", 16)))
    for count in counts:
        residue_count = int(count)
        if residue_count <= 1:
            offset += residue_count
            continue
        section = slice(offset, offset + residue_count)
        hidden = embeddings[section].to(device=logits.device, dtype=logits.dtype)
        local_logits = logits[section]
        local_labels = labels[section]
        local_mask = mask[section]
        selected_count = min(topk, residue_count)
        distance = torch.cdist(hidden, hidden)
        indices = torch.topk(distance, k=selected_count, dim=-1, largest=False).indices
        probability = torch.sigmoid(local_logits).clamp(1e-6, 1.0 - 1e-6)
        maximum = float(getattr(args, "poisson_lambda_max", 20.0))
        rate = -torch.log1p(-probability).clamp_max(maximum)
        region_rate = rate[indices].sum(dim=-1).clamp_max(maximum)
        region_probability = (1.0 - torch.exp(-region_rate)).clamp(1e-6, 1.0 - 1e-6)
        region_logit = torch.logit(region_probability)
        region_labels = local_labels[indices].max(dim=-1).values
        if torch.any(local_mask > 0.5):
            losses.append(masked_bce(region_logit, region_labels, local_mask, pos_weight))
        offset += residue_count
    if not losses:
        return output["logits"].sum() * 0.0
    return torch.stack(losses).mean()


def surface_soft_bce_from_batch(output, batch, pos_weight, args):
    if float(args.surface_soft_loss_weight) <= 0:
        return output["logits"].sum() * 0.0
    losses = []
    atom_offset = 0
    residue_offset = 0
    surface_offset = 0
    for sample_index, residue_count_tensor in enumerate(batch["residue_counts"]):
        atom_count = int(batch["atom_counts"][sample_index].item())
        residue_count = int(residue_count_tensor.item())
        coordinates = batch["X"][atom_offset : atom_offset + atom_count]
        atom_index = (
            batch["atom_to_residue_index"][atom_offset : atom_offset + atom_count] - residue_offset
        )
        labels = batch["y"][residue_offset : residue_offset + residue_count]
        mask = batch["mask"][residue_offset : residue_offset + residue_count]
        cache = batch.get("s1_surface_caches", [None] * len(batch["residue_counts"]))[sample_index]
        if cache is None:
            atom_offset += atom_count
            residue_offset += residue_count
            continue
        surface_position = cache["surface_pos"].to(
            device=output["surface_logits"].device,
            dtype=output["surface_logits"].dtype,
        )
        surface_normal = cache.get("surface_normal")
        if surface_normal is not None:
            surface_normal = surface_normal.to(
                device=surface_position.device,
                dtype=surface_position.dtype,
            )
        surface_count = int(surface_position.shape[0])
        coordinates_float = coordinates.float()
        centers = coordinates_float.new_zeros((residue_count, 3))
        centers.index_add_(0, atom_index.long(), coordinates_float)
        counts = coordinates_float.new_zeros((residue_count, 1))
        counts.index_add_(0, atom_index.long(), coordinates_float.new_ones((atom_count, 1)))
        centers = (centers / counts.clamp_min(1.0)).to(
            device=surface_position.device,
            dtype=surface_position.dtype,
        )
        target = surface_soft_labels(
            labels.to(device=surface_position.device),
            mask.to(device=surface_position.device),
            centers,
            surface_position,
            surface_normal=surface_normal,
            sigma_tangent=args.surface_soft_sigma_tangent,
            sigma_normal=args.surface_soft_sigma_normal,
            max_value=args.surface_soft_max_value,
            cutoff=args.surface_soft_cutoff,
            mode=args.surface_soft_label_mode,
            topk=args.surface_soft_label_topk,
        )
        logits = output["surface_logits"][surface_offset : surface_offset + surface_count]
        losses.append(masked_soft_bce(logits, target, mask=None, pos_weight=pos_weight))
        atom_offset += atom_count
        residue_offset += residue_count
        surface_offset += surface_count
    if not losses:
        return output["logits"].sum() * 0.0
    return torch.stack(losses).mean()


def motif_patch_aux_bce_from_batch(output, batch, pos_weight, args):
    if float(args.motif_patch_aux_loss_weight) <= 0:
        return output["logits"].sum() * 0.0
    losses = []
    labels = batch["y"]
    mask = batch["mask"]
    for logits_key, index_key in (
        ("motif_patch_motif_logits", "motif_patch_motif_primary_idx"),
        ("motif_patch_patch_logits", "motif_patch_patch_primary_idx"),
    ):
        logits = output.get(logits_key)
        indices = output.get(index_key)
        if logits is None or indices is None or logits.numel() == 0 or indices.numel() == 0:
            continue
        indices = indices.to(device=labels.device, dtype=torch.long).clamp(
            0, max(int(labels.numel()) - 1, 0)
        )
        target = labels[indices].to(device=logits.device, dtype=logits.dtype)
        valid = mask[indices].to(device=logits.device, dtype=logits.dtype)
        losses.append(
            masked_bce(
                logits,
                target,
                valid,
                pos_weight.to(device=logits.device, dtype=logits.dtype),
            )
        )
    if not losses:
        return output["logits"].sum() * 0.0
    return torch.stack(losses).mean()


def _positive_components(labels, valid, centers, graph_k, max_components):
    residue_count = int(labels.numel())
    positive = (labels > 0.5) & (valid > 0.5)
    positive_indices = torch.nonzero(positive, as_tuple=False).flatten()
    if positive_indices.numel() == 0:
        return []
    if residue_count < 2:
        return [positive.to(dtype=labels.dtype)]
    selected_count = min(max(int(graph_k), 0), residue_count - 1)
    if selected_count == 0:
        return [positive.to(dtype=labels.dtype)]
    distance = torch.cdist(centers.float(), centers.float())
    distance.fill_diagonal_(float("inf"))
    neighbors = torch.topk(distance, k=selected_count, dim=-1, largest=False).indices
    adjacency = torch.zeros(
        (residue_count, residue_count), device=centers.device, dtype=torch.bool
    )
    rows = torch.arange(residue_count, device=centers.device).unsqueeze(1).expand_as(neighbors)
    adjacency[rows, neighbors] = True
    adjacency = adjacency | adjacency.transpose(0, 1)
    positive_cpu = positive.detach().cpu()
    adjacency_cpu = adjacency.detach().cpu()
    remaining = set(int(index) for index in positive_indices.detach().cpu().tolist())
    components = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        members = [start]
        while stack:
            current = stack.pop()
            for neighbor in torch.nonzero(adjacency_cpu[current], as_tuple=False).flatten().tolist():
                neighbor = int(neighbor)
                if bool(positive_cpu[neighbor]) and neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    members.append(neighbor)
        component = labels.new_zeros((residue_count,))
        component[members] = 1.0
        components.append(component)
    components.sort(key=lambda item: float(item.sum().item()), reverse=True)
    max_components = max(1, int(max_components))
    if len(components) <= max_components:
        return components
    kept = components[:max_components]
    for extra in components[max_components:]:
        extra_indices = torch.nonzero(extra > 0.5, as_tuple=False).flatten()
        extra_center = centers[extra_indices].float().mean(dim=0, keepdim=True)
        distances = []
        for component in kept:
            component_indices = torch.nonzero(component > 0.5, as_tuple=False).flatten()
            component_center = centers[component_indices].float().mean(dim=0, keepdim=True)
            distances.append(torch.linalg.norm(extra_center - component_center).item())
        target_index = int(min(range(len(distances)), key=distances.__getitem__))
        kept[target_index] = torch.maximum(kept[target_index], extra)
    return kept


def _match_patch_masks(predicted, targets, valid):
    if not targets:
        return []
    target = torch.stack(targets, dim=0).to(device=predicted.device, dtype=predicted.dtype)
    valid = valid.to(device=predicted.device, dtype=predicted.dtype).unsqueeze(0)
    prediction = predicted.unsqueeze(1)
    truth = target.unsqueeze(0)
    intersection = (prediction * truth * valid).sum(dim=-1)
    denominator = ((prediction + truth) * valid).sum(dim=-1).clamp_min(1e-6)
    cost = (1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)).detach().cpu()
    if linear_sum_assignment is not None:
        rows, columns = linear_sum_assignment(cost.numpy())
        return list(zip(rows.tolist(), columns.tolist()))
    matches = []
    used_rows = set()
    used_columns = set()
    target_count = int(target.shape[0])
    for flat_index in torch.argsort(cost.flatten()).tolist():
        row = int(flat_index // target_count)
        column = int(flat_index % target_count)
        if row in used_rows or column in used_columns:
            continue
        matches.append((row, column))
        used_rows.add(row)
        used_columns.add(column)
        if len(used_columns) == target_count:
            break
    return matches


def semantic_patch_losses(output, batch, args):
    outputs = output.get("semantic_patch_outputs")
    zero = output["logits"].sum() * 0.0
    result = {"amplitude": zero, "set": zero, "existence": zero, "connectivity": zero}
    if not outputs:
        return result
    amplitude_losses = []
    set_losses = []
    existence_losses = []
    connectivity_losses = []
    residue_offset = 0
    for semantic_output in outputs:
        residue_count = int(semantic_output["region_to_residue"].shape[-1])
        labels = batch["y"][residue_offset : residue_offset + residue_count]
        valid = batch["mask"][residue_offset : residue_offset + residue_count]
        region_weights = semantic_output["region_to_residue"]
        valid_region_weights = region_weights * valid.to(region_weights.dtype).unsqueeze(0)
        region_target = (
            valid_region_weights * labels.to(region_weights.dtype).unsqueeze(0)
        ).sum(dim=-1) / valid_region_weights.sum(dim=-1).clamp_min(1e-8)
        amplitude_losses.append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                semantic_output["region_amplitude_logits"], region_target
            )
        )
        region_existence_logits = semantic_output.get("region_existence_logits")
        if region_existence_logits is not None:
            threshold = max(float(args.semantic_patch_region_existence_threshold), 1e-6)
            region_existence_target = (region_target / threshold).clamp(0.0, 1.0)
            existence_losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    region_existence_logits, region_existence_target
                )
            )
        predicted_masks = semantic_output.get("set_residue_masks")
        existence_logits = semantic_output.get("set_existence_logits")
        if predicted_masks is not None and existence_logits is not None:
            targets = _positive_components(
                labels,
                valid,
                semantic_output["residue_centers"],
                graph_k=args.semantic_patch_component_k,
                max_components=int(predicted_masks.shape[0]),
            )
            matches = _match_patch_masks(predicted_masks, targets, valid)
            existence_target = torch.zeros_like(existence_logits)
            if matches:
                target_stack = torch.stack(targets, dim=0).to(
                    device=predicted_masks.device, dtype=predicted_masks.dtype
                )
                matched_losses = []
                valid_float = valid.to(device=predicted_masks.device, dtype=predicted_masks.dtype)
                for query_index, target_index in matches:
                    existence_target[query_index] = 1.0
                    prediction = predicted_masks[query_index]
                    target = target_stack[target_index]
                    with torch.cuda.amp.autocast(enabled=False):
                        prediction_float = prediction.float()
                        target_float = target.float()
                        valid_float32 = valid_float.float()
                        bce = torch.nn.functional.binary_cross_entropy(
                            prediction_float, target_float, reduction="none"
                        )
                        bce = (bce * valid_float32).sum() / valid_float32.sum().clamp_min(1.0)
                        intersection = (prediction_float * target_float * valid_float32).sum()
                        denominator = (
                            (prediction_float + target_float) * valid_float32
                        ).sum().clamp_min(1e-6)
                        dice = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
                    matched_losses.append(bce + dice)
                set_losses.append(torch.stack(matched_losses).mean())
            existence_losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    existence_logits, existence_target
                )
            )
            query_region_weights = semantic_output["set_region_weights"]
            adjacency = semantic_output["adjacency"].to(
                device=query_region_weights.device, dtype=torch.bool
            )
            non_adjacent = (~adjacency).to(query_region_weights.dtype)
            pair_mass = query_region_weights.unsqueeze(-1) * query_region_weights.unsqueeze(-2)
            connectivity_losses.append(
                (pair_mass * non_adjacent.unsqueeze(0)).sum(dim=(-1, -2)).mean()
            )
        boundary_total_variation = semantic_output.get("semantic_boundary_tv")
        if boundary_total_variation is not None:
            connectivity_losses.append(boundary_total_variation)
        residue_offset += residue_count
    if amplitude_losses:
        result["amplitude"] = torch.stack(amplitude_losses).mean()
    if set_losses:
        result["set"] = torch.stack(set_losses).mean()
    if existence_losses:
        result["existence"] = torch.stack(existence_losses).mean()
    if connectivity_losses:
        result["connectivity"] = torch.stack(connectivity_losses).mean()
    return result
