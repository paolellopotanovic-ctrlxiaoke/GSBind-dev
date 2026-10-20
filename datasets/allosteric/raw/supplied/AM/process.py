import requests
import pandas as pd
import requests
import os
from Bio.PDB import PDBParser, PPBuilder

three_to_one = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

def download_alphafold_structure(uniprot_id, save_dir="pdbs"):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{uniprot_id}.pdb")
    if os.path.exists(file_path):
        return None
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        return None
    else:
        print(f"Failed to download structure for UniProt ID: {uniprot_id}")
        return uniprot_id

def pdb_to_seq(uniprot):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", f"protein/af3/{uniprot}.pdb")
    ppb = PPBuilder()
    peptides = ppb.build_peptides(structure)
    if len(peptides) == 0:
        return ""
    seq = str(peptides[0].get_sequence())
    return seq



import random




# def apply_mut_with_site(seq, mut, uid):
#     i=int(mut[3:-1])-1
#     return (
#         seq[:i]+mut[-1]+seq[i+1:] if 0<=i<len(seq) else seq,
#         f"{uid}_{mut[2]}{i+1}{mut[-1]}"
#     )

# aa="ACDEFGHIKLMNPQRSTVWY"

# def random_mut_with_site(seq, uid):
#     i=random.randrange(len(seq))
#     a=random.choice(aa.replace(seq[i],""))
#     return seq[:i]+a+seq[i+1:], f"{uid}_{seq[i]}{i+1}{a}"

# df=pd.read_csv('AM.csv',sep='\t')
# unis=df['uniprot_id'].unique()
# dic={u:pdb_to_seq(u) for u in unis}
# df['uniprot_wt_sequence']=df['uniprot_id'].map(dic)

# df=df[df.apply(lambda r:r['mut_info'][2]==r['uniprot_wt_sequence'][int(r['mut_info'][3:-1])-1],axis=1)]



# df[['uniprot_positive_sequence','positive_mut_name']] = df.apply(
#     lambda r: pd.Series(apply_mut_with_site(r['uniprot_wt_sequence'], r['mut_info'], r['uniprot_id'])),
#     axis=1
# )


# df[['uniprot_negative_sequence', 'negative_mut_name']] = df.apply(
#     lambda r: pd.Series(random_mut_with_site(r['uniprot_wt_sequence'], r['uniprot_id'])),
#     axis=1
# )


# df = df[['gene_name', 'pdb_id', 'uniprot_id', 'uniprot_wt_sequence', 'uniprot_positive_sequence','positive_mut_name','uniprot_negative_sequence', 'negative_mut_name']]
# df = df[df['uniprot_wt_sequence'].str.len() <= 1500]
# df.to_csv('AM_add.csv',sep='\t',index=False)
df=pd.read_csv('AM_processed.csv',sep='\t')
def balance_group(g):
    pos = g[g['label'] == 1]
    neg = g[g['label'] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return g  # 没法平衡的组，原样保留（或你也可以选择丢掉）
    neg = neg.sample(n=len(pos), random_state=42)
    return pd.concat([pos, neg])

df_balanced = (
    df.groupby('uniprot_id', group_keys=False)
      .apply(balance_group)
      .reset_index(drop=True)
)

df_balanced.to_csv('AM_processed.csv', sep='\t', index=False)


# df=pd.read_csv('AM_add.csv',sep='\t')
# for index,row in df.iterrows():
#     if row['negative_mut_name'] == row['positive_mut_name']:
#         print(111)