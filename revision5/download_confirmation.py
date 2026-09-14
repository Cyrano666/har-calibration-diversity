from pathlib import Path
import urllib.request,json,hashlib,time,concurrent.futures
HERE=Path(__file__).resolve().parent;OUT=HERE/'raw';OUT.mkdir(exist_ok=True)
SOURCES={'pamap2':'http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip','realdisp':'http://archive.ics.uci.edu/static/public/305/realdisp+activity+recognition+dataset.zip'}
def download(item):
 name,url=item;dest=OUT/(name+'.zip');start=time.perf_counter()
 if not dest.exists():
  with urllib.request.urlopen(url,timeout=120) as r,dest.with_suffix('.part').open('wb') as w:
   total=0;last=0
   while b:=r.read(1024*1024):
    w.write(b);total+=len(b)
    if total-last>=64*1024*1024:print(name,total//1024//1024,'MiB',flush=True);last=total
  dest.with_suffix('.part').replace(dest)
 with dest.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 (OUT/(name+'_manifest.json')).write_text(json.dumps({'url':url,'bytes':dest.stat().st_size,'sha256':digest,'seconds':time.perf_counter()-start,'transport_note':'Official UCI HTTP endpoint; HTTPS certificate verification failed as expired on download. Public data only; no credentials transmitted. ZIP CRCs and member hashes are checked during preparation.'},indent=2))
 print(name,'download complete',flush=True)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(2) as pool:list(pool.map(download,SOURCES.items()))
