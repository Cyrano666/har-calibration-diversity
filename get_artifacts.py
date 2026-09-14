"""Install the versioned, checksum-verified numerical artifacts."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,zipfile
ROOT=Path(__file__).resolve().parent
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path);args=ap.parse_args()
    spec=json.loads((ROOT/'artifacts.json').read_text())
    path=args.archive
    if path is None:
        cache=ROOT/'downloads';cache.mkdir(exist_ok=True)
        path=cache/'Online_Resource_1_public.zip'
        if not path.exists():
            partial=path.with_suffix('.part')
            print('Downloading versioned numerical artifacts...',flush=True)
            req=urllib.request.Request(spec['url'],headers={'User-Agent':'JRC-reproduction'})
            with urllib.request.urlopen(req,timeout=60) as response,partial.open('wb') as f:
                while chunk:=response.read(1024*1024):f.write(chunk)
            if partial.stat().st_size!=spec['bytes'] or digest(partial)!=spec['sha256']:
                raise RuntimeError('Downloaded artifact checksum mismatch; partial file retained for inspection')
            partial.replace(path)
    if path.stat().st_size!=spec['bytes'] or digest(path)!=spec['sha256']:
        raise RuntimeError('Artifact checksum does not match artifacts.json')
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read('MANIFEST.json'))
        assert len(z.namelist())==len(set(z.namelist()))
        # Validate every entry and every existing destination before writing.
        for row in manifest:
            dest=(ROOT/row['path']).resolve()
            assert dest.is_relative_to(ROOT), row['path']
            data=z.read(row['path'])
            assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],row['path']
            if dest.exists() and digest(dest)!=row['sha256']:
                raise RuntimeError('Existing file differs; use a clean checkout: '+row['path'])
        for row in manifest:
            dest=ROOT/row['path']
            if not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(row['path']))
    print('Verified and installed',len(manifest),'files. Run python reproduce.py --smoke.',flush=True)
if __name__=='__main__':main()
