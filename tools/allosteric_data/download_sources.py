from concurrent.futures import ThreadPoolExecutor
import json

from common import ROOT, digest, fetch, write_json


SOURCES = {
    'official_AlloBench.csv': 'https://raw.githubusercontent.com/djmaity/allobench/main/AlloBench.csv',
    'official_README.md': 'https://raw.githubusercontent.com/djmaity/allobench/main/README.md',
    'asbench_download.js': 'https://mdl.shsmu.edu.cn/asbench/module/download/download.js',
    'asbench_core.tar.gz': 'https://mdl.shsmu.edu.cn/asbench/datasource/file/AsBench_Core_Set.tar.gz',
    'asbench_diversity.tar.gz': 'https://mdl.shsmu.edu.cn/asbench/datasource/file/ASBench_Core_Diversity_Set.tar.gz',
    'asbench_statistics.xls': 'https://mdl.shsmu.edu.cn/asbench/datasource/file/AsBench_Core_Set_20141111.xls',
    'passer.zip': 'https://passer.smu.edu/static/allorank.zip',
    'casbench_compact.tgz': 'https://biokinet.belozersky.msu.ru/sites/default/data/casbench.tgz',
    'passerrank_README.md': 'https://raw.githubusercontent.com/smu-tao-group/PASSerRank/main/README.md',
    'passerrank_data_source_data_ASD_Release_201909_AS.txt': 'https://raw.githubusercontent.com/smu-tao-group/PASSerRank/main/data/source_data/ASD_Release_201909_AS.txt',
    'alloppi_records.json': 'https://mdl.shsmu.edu.cn/ASD/BrowsePpiMain?start=0&limit=10000&page=1',
    'alloppi_detail.js': 'https://mdl.shsmu.edu.cn/ASD/module/alloppi/js/alloppi_detail.js',
    'alloppi_record.js': 'https://mdl.shsmu.edu.cn/ASD/module/alloppi/js/alloppi_record.js',
}


def main():
    folder = ROOT / 'raw/external'
    def download(item):
        name,url = item
        path = folder / name
        fetch(url,path)
        return dict(path=str(path.relative_to(ROOT)),url=url,sha256=digest(path),bytes=path.stat().st_size)
    with ThreadPoolExecutor(max_workers=6) as executor:
        records = list(executor.map(download,SOURCES.items()))
    entries = json.loads((folder / 'alloppi_records.json').read_text())['data']
    jobs = []
    for row in entries:
        identifier = row['main_id']
        jobs.extend([(f'alloppi_details/{identifier}.json',f'https://mdl.shsmu.edu.cn/ASD/BrowsePpiDetail?main_id={identifier}'),
                     (f'alloppi_modulators/{identifier}.json',f'https://mdl.shsmu.edu.cn/ASD/BrowsePpiModulator?main_id={identifier}&draw=1&start=0&length=10000')])
    with ThreadPoolExecutor(max_workers=6) as executor:
        records.extend(executor.map(download,jobs))
    write_json(ROOT / 'external_download_manifest.json',records)
    print('source files',len(records),flush=True)


if __name__ == '__main__':
    main()
