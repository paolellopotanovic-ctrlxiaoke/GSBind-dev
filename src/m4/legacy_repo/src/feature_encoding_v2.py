import numpy as np
import torch as pt

from src.data_encoding import (
    std_elements,
    std_resnames,
    std_names,
    onehot,
    encode_structure,
    encode_features,
)


PHYS_FEATURE_NAMES = np.array([
    "is_backbone",
    "is_sidechain",
    "is_ca",
    "is_hetero",
    "is_protein",
    "is_nucleic_acid",
    "is_ligand",
    "is_ion",
    "is_lipid",
    "res_charge_pos",
    "res_charge_neg",
    "res_polar",
    "res_hydrophobic",
    "res_aromatic",
    "res_sulfur",
    "atom_donor_like",
    "atom_acceptor_like",
    "atom_is_metal",
    "atom_is_halogen",
    "atom_is_sulfur",
    "atom_is_phosphorus",
    "vdw_radius_norm",
    "covalent_radius_norm",
    "atomic_mass_norm",
])

EDGE_FEATURE_NAMES = np.array(
    [f"distance_rbf_{i:02d}" for i in range(16)]
    + [
        "distance_norm",
        "same_residue",
        "same_chain",
        "inter_chain",
        "residue_gap_1",
        "residue_gap_2_4",
        "residue_gap_ge5",
        "sequential_backbone_candidate",
        "covalent_candidate",
    ]
)

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
POSITIVE_RESIDUES = {"LYS", "ARG", "HIS"}
NEGATIVE_RESIDUES = {"ASP", "GLU"}
POLAR_RESIDUES = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS"}
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
SULFUR_RESIDUES = {"CYS", "MET"}
PROTEIN_RESIDUES = set(std_resnames[:20].astype(str))
NUCLEIC_RESIDUES = {"A", "U", "G", "C", "DA", "DT", "DG", "DC"}
ION_RESIDUES = {
    "MG",
    "ZN",
    "CL",
    "CA",
    "NA",
    "MN",
    "K",
    "IOD",
    "CD",
    "CU",
    "FE",
    "NI",
    "SR",
    "BR",
    "CO",
    "HG",
}
LIPID_RESIDUES = {"PLM", "CLR", "CDL", "RET"}
METAL_ELEMENTS = {"MG", "ZN", "CA", "NA", "MN", "K", "CD", "CU", "FE", "NI", "SR", "CO", "HG"}
HALOGEN_ELEMENTS = {"F", "CL", "BR", "I"}

VDW_RADIUS = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "SE": 1.90,
    "BR": 1.85,
    "I": 1.98,
    "MG": 1.73,
    "ZN": 1.39,
    "FE": 1.32,
    "CA": 2.31,
    "NA": 2.27,
    "K": 2.75,
}

COVALENT_RADIUS = {
    "H": 0.31,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "SE": 1.20,
    "BR": 1.20,
    "I": 1.39,
    "MG": 1.41,
    "ZN": 1.22,
    "FE": 1.24,
    "CA": 1.76,
    "NA": 1.66,
    "K": 2.03,
}

ATOMIC_MASS = {
    "H": 1.008,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "NA": 22.990,
    "MG": 24.305,
    "P": 30.974,
    "S": 32.06,
    "CL": 35.45,
    "K": 39.098,
    "CA": 40.078,
    "MN": 54.938,
    "FE": 55.845,
    "CO": 58.933,
    "NI": 58.693,
    "CU": 63.546,
    "ZN": 65.38,
    "SE": 78.971,
    "BR": 79.904,
    "SR": 87.62,
    "CD": 112.414,
    "I": 126.904,
    "HG": 200.592,
}


def _upper_array(values):
    return np.char.upper(values.astype(str))


def _index_from_vocab(values, vocab):
    values = _upper_array(values)
    vocab = _upper_array(np.asarray(vocab))
    out = np.full(values.shape, len(vocab), dtype=np.int64)
    lookup = {v: i for i, v in enumerate(vocab)}
    for i, value in enumerate(values):
        out[i] = lookup.get(value, len(vocab))
    return out


def element_indices(elements):
    return _index_from_vocab(elements, std_elements)


def atom_name_indices(names):
    return _index_from_vocab(names, std_names)


def encode_phys_features(structure, device=pt.device("cpu")):
    names = _upper_array(structure["name"])
    elements = _upper_array(structure["element"])
    resnames = _upper_array(structure["resname"])
    het_flags = structure["het_flag"].astype(str)

    features = np.zeros((len(names), len(PHYS_FEATURE_NAMES)), dtype=np.float32)
    atom_names = [str(x).strip().upper() for x in names]
    elem_names = [str(x).strip().upper() for x in elements]
    residue_names = [str(x).strip().upper() for x in resnames]

    is_backbone = np.array([name in BACKBONE_ATOMS for name in atom_names], dtype=bool)
    is_ca = np.array([name == "CA" for name in atom_names], dtype=bool)
    is_protein = np.array([rn in PROTEIN_RESIDUES for rn in residue_names], dtype=bool)
    is_nucleic = np.array([rn in NUCLEIC_RESIDUES for rn in residue_names], dtype=bool)
    is_ion = np.array([rn in ION_RESIDUES or el in METAL_ELEMENTS for rn, el in zip(residue_names, elem_names)], dtype=bool)
    is_lipid = np.array([rn in LIPID_RESIDUES for rn in residue_names], dtype=bool)
    is_ligand = ~(is_protein | is_nucleic | is_ion | is_lipid)

    bool_features = {
        "is_backbone": is_backbone,
        "is_sidechain": is_protein & ~is_backbone,
        "is_ca": is_ca,
        "is_hetero": het_flags == "H",
        "is_protein": is_protein,
        "is_nucleic_acid": is_nucleic,
        "is_ligand": is_ligand,
        "is_ion": is_ion,
        "is_lipid": is_lipid,
        "res_charge_pos": np.array([rn in POSITIVE_RESIDUES for rn in residue_names]),
        "res_charge_neg": np.array([rn in NEGATIVE_RESIDUES for rn in residue_names]),
        "res_polar": np.array([rn in POLAR_RESIDUES for rn in residue_names]),
        "res_hydrophobic": np.array([rn in HYDROPHOBIC_RESIDUES for rn in residue_names]),
        "res_aromatic": np.array([rn in AROMATIC_RESIDUES for rn in residue_names]),
        "res_sulfur": np.array([rn in SULFUR_RESIDUES for rn in residue_names]),
        "atom_donor_like": np.array([el in {"N", "O", "S"} for el in elem_names]),
        "atom_acceptor_like": np.array([el in {"N", "O", "S", "P"} for el in elem_names]),
        "atom_is_metal": np.array([el in METAL_ELEMENTS for el in elem_names]),
        "atom_is_halogen": np.array([el in HALOGEN_ELEMENTS for el in elem_names]),
        "atom_is_sulfur": np.array([el == "S" for el in elem_names]),
        "atom_is_phosphorus": np.array([el == "P" for el in elem_names]),
    }

    name_to_col = {name: i for i, name in enumerate(PHYS_FEATURE_NAMES)}
    for key, value in bool_features.items():
        features[:, name_to_col[key]] = value.astype(np.float32)

    vdw = np.array([VDW_RADIUS.get(el, 1.70) for el in elem_names], dtype=np.float32)
    cov = np.array([COVALENT_RADIUS.get(el, 0.76) for el in elem_names], dtype=np.float32)
    mass = np.array([ATOMIC_MASS.get(el, 12.011) for el in elem_names], dtype=np.float32)
    features[:, name_to_col["vdw_radius_norm"]] = np.clip(vdw / 2.8, 0.0, 1.0)
    features[:, name_to_col["covalent_radius_norm"]] = np.clip(cov / 2.1, 0.0, 1.0)
    features[:, name_to_col["atomic_mass_norm"]] = np.clip(mass / 200.0, 0.0, 1.0)

    return pt.from_numpy(features).to(device)


def encode_features_v2(structure, device=pt.device("cpu")):
    qe, qr, qn = encode_features(structure, device=device)
    q_phys = encode_phys_features(structure, device=device)
    return qe, qr, qn, q_phys


def structure_to_data_v2(structure, num_nn=64, device=pt.device("cpu")):
    from src.data_encoding import extract_topology

    X, M = encode_structure(structure, device=device)
    qe, qr, qn, q_phys = encode_features_v2(structure, device=device)
    ids_topk = extract_topology(X, num_nn)[0]
    q0 = pt.cat([qe, qr, qn, q_phys], dim=1)
    return X, ids_topk, q0, M


config_encoding_v2 = {
    "std_elements": std_elements,
    "std_resnames": std_resnames,
    "std_names": std_names,
    "phys_feature_names": PHYS_FEATURE_NAMES,
    "edge_feature_names": EDGE_FEATURE_NAMES,
}
