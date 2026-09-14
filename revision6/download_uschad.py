from pathlib import Path
import urllib.request,hashlib,json,zipfile,time
HERE=Path(__file__).resolve().parent;OUT=HERE/'raw';OUT.mkdir(exist_ok=True)
path=OUT/'USC-HAD.zip';url='https://sipi.usc.edu/had/USC-HAD.zip'
if not path.exists():
    part=path.with_suffix('.part');start=time.time();last=start
    with urllib.request.urlopen(url,timeout=60) as response,part.open('wb') as f:
        total=response.headers.get('Content-Length');n=0
        while True:
            block=response.read(1024*1024)
            if not block:break
            f.write(block);n+=len(block)
            if time.time()-last>20:print(n,total,'bytes',flush=True);last=time.time()
    part.replace(path)
with zipfile.ZipFile(path) as z:
    bad=z.testzip();assert bad is None
    print('Archive entries:',len(z.namelist()),flush=True)
    print('\n'.join(z.namelist()[:18]),flush=True)
    for name in z.namelist():
        if any(q in name.lower() for q in ['readme','license','copyright']) and not name.endswith('/'):
            text=z.read(name).decode(errors='replace');print(name,text[:10000],flush=True)
manifest=dict(url=url,bytes=path.stat().st_size,sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),crc='passed',stage='No model outcomes inspected')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
