from collections import Counter
import io
import pickle
import pickletools
import zipfile

from common import ROOT, write_json


def pure_containers(content):
    allowed = {'PROTO','FRAME','EMPTY_LIST','EMPTY_TUPLE','EMPTY_DICT','MARK','APPENDS','APPEND','SETITEMS','SETITEM','TUPLE','TUPLE1','TUPLE2','TUPLE3','BINPUT','LONG_BINPUT','BINGET','LONG_BINGET','MEMOIZE','BININT','BININT1','BININT2','LONG','LONG1','LONG4','INT','FLOAT','BINFLOAT','NONE','NEWTRUE','NEWFALSE','STOP','BINUNICODE','SHORT_BINUNICODE','UNICODE'}
    operations = Counter(operation.name for operation,_,_ in pickletools.genops(content))
    if set(operations)-allowed:
        raise ValueError('Non-container opcodes refused: '+str(set(operations)-allowed))
    class RefuseGlobals(pickle.Unpickler):
        def find_class(self,module,name):
            raise ValueError('Global loading forbidden')
    return RefuseGlobals(io.BytesIO(content)).load(),dict(operations)


def main():
    summary = {}
    with zipfile.ZipFile(ROOT / 'raw/external/passer.zip') as archive:
        for dataset in ['asd','casbench']:
            features,feature_ops = pure_containers(archive.read(f'allorank/{dataset}/features.pkl'))
            labels,label_ops = pure_containers(archive.read(f'allorank/{dataset}/labels.pkl'))
            if len(features) != len(labels):
                raise ValueError('Feature/label group mismatch')
            summary[dataset] = dict(protein_or_pocket_groups=len(features),pockets=sum(len(group) for group in labels),positive_pockets=sum(sum(group) for group in labels),
                feature_container=type(features).__name__,feature_opcodes=feature_ops,label_opcodes=label_ops,
                identity_mapping_present=False,GSbind_residue_evaluation_possible_from_these_files_alone=False)
    write_json(ROOT / 'passer_prepared_archive_audit.json',summary)
    print(summary,flush=True)


if __name__ == '__main__':
    main()
