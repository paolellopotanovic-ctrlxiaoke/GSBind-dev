import torch

try:
    from scipy.optimize import linear_sum_assignment
except ModuleNotFoundError:
    linear_sum_assignment = None

def batch_sample_weights(batch, *, device, dtype):
    weights = batch.get("sample_weights")
    if weights is None:
        weights = torch.ones(len(batch.get("residue_counts", ())), dtype=dtype, device=device)
    weights = weights.to(device=device, dtype=dtype)
    if weights.ndim != 1 or int(weights.numel()) != len(batch.get("residue_counts", ())):
        raise ValueError("sample_weights must contain one value per sample")
    if not torch.all(torch.isfinite(weights)) or torch.any(weights <= 0):
        raise ValueError("sample_weights must be finite and strictly positive")
    return weights

def weighted_sample_mean(losses, weights, zero):
    if not losses:
        return zero
    values = torch.stack(losses)
    sample_weights = torch.stack(
        [torch.as_tensor(weight, device=values.device, dtype=values.dtype) for weight in weights]
    )
    return (values * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)

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
    cost = torch.nan_to_num(
        (1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)).detach().float().cpu(),
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    )
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
    zero = output["final_logits"].sum() * 0.0
    result = {"amplitude": zero, "set": zero, "existence": zero, "connectivity": zero}
    if not outputs:
        return result
    amplitude_losses = []
    set_losses = []
    existence_losses = []
    connectivity_losses = []
    amplitude_weights = []
    set_weights = []
    existence_weights = []
    connectivity_weights = []
    sample_weights = batch_sample_weights(
        batch,
        device=output["final_logits"].device,
        dtype=output["final_logits"].dtype,
    )
    residue_offset = 0
    for sample_index, semantic_output in enumerate(outputs):
        residue_count = int(semantic_output["region_to_residue"].shape[-1])
        labels = batch["y"][residue_offset : residue_offset + residue_count]
        valid = batch["mask"][residue_offset : residue_offset + residue_count]
        region_weights = torch.nan_to_num(
            semantic_output["region_to_residue"], nan=0.0, posinf=0.0, neginf=0.0
        )
        valid_region_weights = region_weights * valid.to(region_weights.dtype).unsqueeze(0)
        region_target = (
            valid_region_weights * labels.to(region_weights.dtype).unsqueeze(0)
        ).sum(dim=-1) / valid_region_weights.sum(dim=-1).clamp_min(1e-8)
        region_target = torch.nan_to_num(
            region_target, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        region_amplitude_logits = torch.nan_to_num(
            semantic_output["region_amplitude_logits"], nan=0.0, posinf=20.0, neginf=-20.0
        )
        amplitude_losses.append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                region_amplitude_logits, region_target
            )
        )
        amplitude_weights.append(sample_weights[sample_index])
        region_existence_logits = semantic_output.get("region_existence_logits")
        if region_existence_logits is not None:
            region_existence_logits = torch.nan_to_num(
                region_existence_logits, nan=0.0, posinf=20.0, neginf=-20.0
            )
            threshold = max(float(args.semantic_patch_region_existence_threshold), 1e-6)
            region_existence_target = (region_target / threshold).clamp(0.0, 1.0)
            existence_losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    region_existence_logits, region_existence_target
                )
            )
            existence_weights.append(sample_weights[sample_index])
        predicted_masks = semantic_output.get("set_residue_masks")
        existence_logits = semantic_output.get("set_existence_logits")
        if predicted_masks is not None and existence_logits is not None:
            predicted_masks = torch.nan_to_num(
                predicted_masks, nan=0.5, posinf=1.0 - 1e-4, neginf=1e-4
            ).clamp(1e-4, 1.0 - 1e-4)
            existence_logits = torch.nan_to_num(
                existence_logits, nan=0.0, posinf=20.0, neginf=-20.0
            )
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
                set_weights.append(sample_weights[sample_index])
            existence_losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    existence_logits, existence_target
                )
            )
            existence_weights.append(sample_weights[sample_index])
            query_region_weights = torch.nan_to_num(
                semantic_output["set_region_weights"], nan=0.0, posinf=0.0, neginf=0.0
            )
            adjacency = semantic_output["adjacency"].to(
                device=query_region_weights.device, dtype=torch.bool
            )
            non_adjacent = (~adjacency).to(query_region_weights.dtype)
            pair_mass = query_region_weights.unsqueeze(-1) * query_region_weights.unsqueeze(-2)
            connectivity_losses.append(
                (pair_mass * non_adjacent.unsqueeze(0)).sum(dim=(-1, -2)).mean()
            )
            connectivity_weights.append(sample_weights[sample_index])
        boundary_total_variation = semantic_output.get("semantic_boundary_tv")
        if boundary_total_variation is not None:
            connectivity_losses.append(
                torch.nan_to_num(boundary_total_variation, nan=0.0, posinf=0.0, neginf=0.0)
            )
            connectivity_weights.append(sample_weights[sample_index])
        residue_offset += residue_count
    result["amplitude"] = weighted_sample_mean(amplitude_losses, amplitude_weights, zero)
    result["set"] = weighted_sample_mean(set_losses, set_weights, zero)
    result["existence"] = weighted_sample_mean(existence_losses, existence_weights, zero)
    result["connectivity"] = weighted_sample_mean(
        connectivity_losses, connectivity_weights, zero
    )
    return result
