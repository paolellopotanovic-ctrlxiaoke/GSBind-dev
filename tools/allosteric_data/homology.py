import subprocess
import os
import shutil

import pandas as pd

from common import LEGACY, ROOT, write_json


def main():
    folder = ROOT / 'homology'
    folder.mkdir(exist_ok=True)
    manifest = pd.read_csv(ROOT / 'manifest_with_external.csv')
    query = folder / 'query.fasta'
    query.write_text(''.join('>' + row.uniprot + '\n' + row.sequence + '\n' for row in manifest.itertuples()))
    binary = os.environ.get('MMSEQS_BIN') or shutil.which('mmseqs')
    if not binary:
        raise FileNotFoundError('Install MMseqs2 or set MMSEQS_BIN')
    commands = []
    for name, target in [('query', query), ('e11', LEGACY / 'homology/e11_training.fasta')]:
        output = folder / (name + '_hits.tsv')
        command = [str(binary), 'easy-search', str(query), str(target), str(output), str(folder / ('tmp_'+name)), '--min-seq-id', '0.30', '-c', '0.0', '--cov-mode', '0', '-s', '7.5', '--max-seqs', '100000', '--threads', '6', '--format-output', 'query,target,fident,qcov,tcov,evalue,bits']
        commands.append(command)
        if not output.exists():
            with (folder / (name+'.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    columns = ['query', 'target', 'identity', 'qcov', 'tcov', 'evalue', 'bits']
    hits = pd.read_csv(folder / 'query_hits.tsv', sep='\t', names=columns)
    paired = hits[hits.identity.ge(.3) & hits.qcov.ge(.8) & hits.tcov.ge(.8)]
    parent = {identifier: identifier for identifier in manifest.uniprot}
    def representative(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier
    for row in paired.itertuples():
        left, right = representative(row.query), representative(row.target)
        parent[max(left, right)] = min(left, right)
    manifest['component'] = [representative(identifier) for identifier in manifest.uniprot]
    training = pd.read_csv(folder / 'e11_hits.tsv', sep='\t', names=columns)
    for label, mask in [('both80', training.qcov.ge(.8) & training.tcov.ge(.8)), ('either80', training.qcov.ge(.8) | training.tcov.ge(.8))]:
        matching = training[training.identity.ge(.3) & mask]
        manifest['e11_'+label+'_hit'] = manifest.uniprot.isin(matching['query'])
    manifest[['uniprot', 'component', 'e11_both80_hit', 'e11_either80_hit']].to_csv(folder / 'membership.csv', index=False)
    write_json(folder / 'contract.json', dict(commands=commands, proteins=len(manifest), components=int(manifest.component.nunique()), identity=.3, qcov=.8, tcov=.8))
    print('homology', len(manifest), manifest.component.nunique(), flush=True)


if __name__ == '__main__':
    main()
