import numpy as np

res3to1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
}


def clean_structure(structure, rm_wat=True):
    # mask for water, hydrogens and deuterium
    m_wat = (structure["resname"] == "HOH")
    m_h = (structure["element"] == "H")
    m_d = (structure["element"] == "D")
    m_hwat = (structure["resname"] == "DOD")

    if rm_wat:
        # remove water
        mask = ((~m_wat) & (~m_h) & (~m_d) & (~m_hwat))
    else:
        # keep but tag water
        mask = ((~m_h) & (~m_d) & (~m_hwat))
        structure["resid"][m_wat] = -999

    # filter structure atoms
    structure = {key:structure[key][mask] for key in structure}

    # find changes due to chain
    chains = structure["chain_name"]
    ids_chains = np.where(np.array(chains).reshape(-1,1) == np.unique(chains).reshape(1,-1))[1]
    delta_chains = np.abs(np.sign(np.concatenate([[0], np.diff(ids_chains)])))

    # find changes due to inertion code
    icodes = structure["icode"]
    ids_icodes = np.where(np.array(icodes).reshape(-1,1) == np.unique(icodes).reshape(1,-1))[1]
    delta_icodes = np.abs(np.sign(np.concatenate([[0], np.diff(ids_icodes)])))

    # find changes due to resids
    resids = structure["resid"]
    delta_resids = np.abs(np.sign(np.concatenate([[0], np.diff(resids)])))

    # renumber resids
    resids = np.cumsum(np.sign(delta_chains + delta_resids + delta_icodes)) + 1

    # update resids
    structure['resid'] = resids

    # remove uncessary icodes
    structure.pop("icode")

    # return process structure
    return structure
