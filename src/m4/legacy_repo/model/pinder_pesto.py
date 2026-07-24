import torch as pt

from src.model_operations import StatePoolLayer, StateUpdateLayer, unpack_state_features


STATE_TO_ID = {"holo": 0, "apo": 1, "af2": 2}


class PinderPeSTo(pt.nn.Module):
    """PeSTo-compatible encoder with a binary future-interface head."""

    def __init__(self, config, state_embedding_dim=0, state_mode="none"):
        super().__init__()
        if state_mode not in {"none", "concat", "concat_warmup", "film", "film_warmup"}:
            raise ValueError(f"Unsupported state_mode: {state_mode}")
        self.state_mode = state_mode
        self.state_embedding_dim = int(state_embedding_dim)
        self.state_feature_dim = 2 * config["dm"]["N0"]
        self.register_buffer("state_alpha", pt.tensor(1.0, dtype=pt.float32), persistent=False)
        self.em = pt.nn.Sequential(
            pt.nn.Linear(config["em"]["N0"], config["em"]["N1"]),
            pt.nn.ELU(),
            pt.nn.Linear(config["em"]["N1"], config["em"]["N1"]),
            pt.nn.ELU(),
            pt.nn.Linear(config["em"]["N1"], config["em"]["N1"]),
        )
        self.sum = pt.nn.Sequential(*[StateUpdateLayer(layer_params) for layer_params in config["sum"]])
        self.spl = StatePoolLayer(config["spl"]["N0"], config["spl"]["N1"], config["spl"]["Nh"])

        if self.state_mode != "none" and self.state_embedding_dim <= 0:
            raise ValueError("state_embedding_dim must be positive when state_mode is not none")
        if self.state_mode != "none":
            self.state_embedding = pt.nn.Embedding(len(STATE_TO_ID), self.state_embedding_dim)
        else:
            self.state_embedding = None
        if self.state_mode in {"film", "film_warmup"}:
            self.state_film = pt.nn.Sequential(
                pt.nn.Linear(self.state_embedding_dim, self.state_feature_dim),
                pt.nn.ELU(),
                pt.nn.Linear(self.state_feature_dim, 2 * self.state_feature_dim),
            )
            pt.nn.init.zeros_(self.state_film[-1].weight)
            pt.nn.init.zeros_(self.state_film[-1].bias)
        else:
            self.state_film = None

        concat_dim = self.state_embedding_dim if self.state_mode in {"concat", "concat_warmup"} else 0
        decoder_in = self.state_feature_dim + concat_dim
        self.dm = pt.nn.Sequential(
            pt.nn.Linear(decoder_in, config["dm"]["N1"]),
            pt.nn.ELU(),
            pt.nn.Linear(config["dm"]["N1"], config["dm"]["N1"]),
            pt.nn.ELU(),
            pt.nn.Linear(config["dm"]["N1"], 1),
        )

    def set_state_alpha(self, alpha):
        self.state_alpha.fill_(float(alpha))

    def forward(self, X, ids_topk, q0, M, state_ids=None):
        q = self.em.forward(q0)
        p0 = pt.zeros((q.shape[0] + 1, X.shape[1], q.shape[1]), device=X.device, dtype=q.dtype)
        q, ids_topk, D_nn, R_nn = unpack_state_features(X, ids_topk, q)
        qa, pa, _, _, _ = self.sum.forward((q, p0, ids_topk, D_nn, R_nn))
        qr, pr = self.spl.forward(qa[1:], pa[1:], M)
        zr = pt.cat([qr, pt.norm(pr, dim=1)], dim=1)

        if self.state_mode != "none":
            if state_ids is None:
                raise ValueError("state_ids is required when state_mode is not none")
            z_state = self.state_embedding(state_ids.long().to(zr.device))
            alpha = self.state_alpha.to(device=zr.device, dtype=zr.dtype)
            if self.state_mode == "concat_warmup":
                z_state = z_state * alpha
            if self.state_mode in {"concat", "concat_warmup"}:
                zr = pt.cat([zr, z_state], dim=1)
            elif self.state_mode in {"film", "film_warmup"}:
                gamma_beta = self.state_film(z_state)
                gamma, beta = pt.chunk(gamma_beta, 2, dim=1)
                if self.state_mode == "film_warmup":
                    gamma = gamma * alpha
                    beta = beta * alpha
                zr = zr * (1.0 + gamma) + beta

        return self.dm.forward(zr).squeeze(1)


def load_pesto_encoder_weights(model, checkpoint_path, map_location="cpu"):
    """Load only PeSTo encoder/pool weights, leaving the binary head fresh."""

    checkpoint = pt.load(checkpoint_path, map_location=map_location)
    model_state = model.state_dict()
    loaded = {}
    skipped = []
    for key, value in checkpoint.items():
        if key.startswith("dm."):
            skipped.append(key)
            continue
        if key in model_state and model_state[key].shape == value.shape:
            loaded[key] = value
        else:
            skipped.append(key)
    model_state.update(loaded)
    model.load_state_dict(model_state)
    return {"loaded": sorted(loaded), "skipped": sorted(skipped)}
