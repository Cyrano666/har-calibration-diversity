from pathlib import Path
import sys,io,zipfile,re,json,hashlib
HERE=Path(__file__).resolve().parent;V5=HERE.parent/'revision5'
sys.path[:0]=[str(V5/'fitdeps'),str(V5)]
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from prepare_confirmation import statistics
OUT=HERE/'data';OUT.mkdir(exist_ok=True)
assert not list((HERE/'results/predictions').glob('uschad*.npz'))
x=[];y=[];g=[];ids=[];qc=[]
with zipfile.ZipFile(HERE/'raw/USC-HAD.zip') as archive:
 readme=archive.read('USC-HAD/Readme.txt').decode();assert '100Hz' in readme and 'acc_x' in readme
 (OUT/'USC_original_readme.txt').write_text(readme)
 files=sorted(n for n in archive.namelist() if n.endswith('.mat'))
 for name in files:
  match=re.search(r'Subject(\d+)/a(\d+)t(\d+)\.mat$',name);assert match
  user,activity,trial=map(int,match.groups());raw=archive.read(name);z=loadmat(io.BytesIO(raw));a=z['sensor_readings']
  assert a.ndim==2 and a.shape[1]==6 and 1<=activity<=12
  if not qc:print('MAT keys',list(z),'shape',a.shape,flush=True)
  before=len(x);bad=0
  for start in range(0,len(a)-499,500):
   window=a[start:start+500,:3]
   if not np.isfinite(window).all():bad+=1;continue
   value=resample_poly(window,32,125,axis=0,padtype='line').T.astype('float32');assert value.shape==(3,128)
   x.append(value);y.append(activity-1);g.append(user);ids.append(f'{name}:{start}:{start+500}')
  qc.append(dict(file=name,subject=user,activity=activity,trial=trial,rows=len(a),windows=len(x)-before,rejected_nonfinite=bad,sha256=hashlib.sha256(raw).hexdigest()))
X=np.asarray(x);y=np.asarray(y,dtype='int32');g=np.asarray(g,dtype='int32');ids=np.asarray(ids)
counts={int(u):int(np.sum(g==u)) for u in np.unique(g)};eligible=[u for u,n in counts.items() if n>=120]
assert len(eligible)==14 and len(np.unique(y))==12
Z=statistics(X);assert X.shape[1:]==(3,128) and np.isfinite(Z).all()
np.savez_compressed(OUT/'uschad.npz',X=X,Z=Z,y=y,g=g,ids=ids)
meta=dict(dataset='uschad',windows=len(y),subjects=len(eligible),classes=12,subject_counts=counts,eligible_subjects=eligible,
 sensor='front right hip acceleration',zero_based_signal_columns=[0,1,2],duration_seconds=5,sampling_hz=100,
 shape=list(X.shape),feature_count=35,quality=qc,signal_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
 source_url='https://sipi.usc.edu/had/',stage='Preprocessing metadata before predictive outcomes; method locked already',
 redistribution='Original data publicly downloadable for research; no explicit CC license found on landing page/readme. Package downloader and derived results, not raw signals.')
(OUT/'uschad_metadata.json').write_text(json.dumps(meta,indent=2));print(len(y),'windows',counts,flush=True)
