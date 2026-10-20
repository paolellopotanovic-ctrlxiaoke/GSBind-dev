import torch as pt

from .operations import StateUpdateLayer, StatePoolLayer, unpack_state_features


class AtomEncoder(pt.nn.Module):
    def __init__(self, config):
        super(AtomEncoder, self).__init__()
        self.atom_feature_mlp = pt.nn.Sequential(
            pt.nn.Linear(config['atom_features']['N0'], config['atom_features']['N1']),
            pt.nn.ELU(),
            pt.nn.Linear(config['atom_features']['N1'], config['atom_features']['N1']),
            pt.nn.ELU(),
            pt.nn.Linear(config['atom_features']['N1'], config['atom_features']['N1']),
        )
        self.state_updates = pt.nn.Sequential(*[
            StateUpdateLayer(layer_params)
            for layer_params in config['state_updates']
        ])

        self.atom_to_residue_pool = StatePoolLayer(
            config['residue_pool']['N0'],
            config['residue_pool']['N1'],
            config['residue_pool']['Nh'],
        )

        self.residue_decoder = pt.nn.Sequential(
            pt.nn.Linear(2*config['residue_decoder']['N0'], config['residue_decoder']['N1']),
            pt.nn.ELU(),
            pt.nn.Linear(config['residue_decoder']['N1'], config['residue_decoder']['N1']),
            pt.nn.ELU(),
            pt.nn.Linear(config['residue_decoder']['N1'], config['residue_decoder']['N2']),
        )

    def residue_states(self, X, ids_topk, q0, M):
        q = self.atom_feature_mlp.forward(q0)

        p0 = pt.zeros((q.shape[0]+1, X.shape[1], q.shape[1]), device=X.device)

        q, ids_topk, D_nn, R_nn = unpack_state_features(X, ids_topk, q)

        qa, pa, _, _, _ = self.state_updates.forward(
            (q, p0, ids_topk, D_nn, R_nn)
        )

        return self.atom_to_residue_pool.forward(qa[1:], pa[1:], M)

    def decode_residues(self, qr, pr):
        zr = pt.cat([qr, pt.norm(pr, dim=1)], dim=1)
        return self.residue_decoder.forward(zr)

    def forward(self, X, ids_topk, q0, M):
        qr, pr = self.residue_states(X, ids_topk, q0, M)
        z = self.decode_residues(qr, pr)

        return z
