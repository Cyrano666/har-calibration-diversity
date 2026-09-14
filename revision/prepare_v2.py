from pathlib import Path
import sys,io,zipfile,json,re,hashlib
BASE=Path(__file__).resolve().parent
# Preparation also works with the previous analysis runtime.
sys.path.insert(0,str(BASE.parent/'pydeps'))
import numpy as np,pandas as pd
D=BASE/'data';D.mkdir(exist_ok=True)
def archive(name):
 p=BASE/'raw'/(name+'.zip')
 return p if p.exists() else BASE.parent/'data'/(name+'.zip')
def resample(x,length=128):
 return np.stack([np.interp(np.linspace(0,len(x)-1,length),np.arange(len(x)),x[:,j]) for j in range(3)])
def statistics(x):
 mag=np.sqrt((x*x).sum(1))[:,None,:];a=np.concatenate([x,mag],axis=1)
 feats=[a.mean(2),a.std(2),a.min(2),a.max(2),np.sqrt((a*a).mean(2)),np.abs(np.diff(a,axis=2)).mean(2),np.quantile(a,.25,axis=2),np.quantile(a,.75,axis=2)]
 for i,j in [(0,1),(0,2),(1,2)]:
  u=x[:,i]-x[:,i].mean(1)[:,None];v=x[:,j]-x[:,j].mean(1)[:,None];den=np.sqrt((u*u).sum(1)*(v*v).sum(1));feats.append(np.divide((u*v).sum(1),den,out=np.zeros(len(x)),where=den>1e-12)[:,None])
 return np.concatenate(feats,axis=1).astype('float32')
def save(name,x,y,g,ids,details):
 X=np.asarray(x,dtype='float32');y=np.array(y,dtype='int32');g=np.array(g,dtype='int32');assert X.shape[1:]==(3,128) and np.isfinite(X).all()
 Z=statistics(X);np.savez_compressed(D/(name+'.npz'),X=X,Z=Z,y=y,g=g,ids=np.array(ids))
 counts={str(int(u)):int((g==u).sum()) for u in np.unique(g)}
 meta=dict(dataset=name,windows=len(y),subjects=len(counts),classes=len(np.unique(y)),signal_shape=list(X.shape),feature_count=Z.shape[1],subject_counts=counts,class_counts=np.bincount(y).tolist(),min_subject_class_count=int(min(np.sum((g==u)&(y==c)) for u in np.unique(g) for c in np.unique(y))),signal_sha256=hashlib.sha256(X.tobytes()).hexdigest(),details=details)
 (D/(name+'_metadata.json')).write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if '--force-har' in sys.argv or not (D/'har.npz').exists():
 z=zipfile.ZipFile(archive('har'));z=zipfile.ZipFile(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith('.zip')))))
 def read(s):return z.read(next(n for n in z.namelist() if n.endswith(s) and not n.startswith('__MACOSX')))
 xx=[];yy=[];gg=[];ids=[]
 for split in ['train','test']:
  X=np.stack([np.loadtxt(io.BytesIO(read(f'/{split}/Inertial Signals/total_acc_{axis}_{split}.txt'))) for axis in ['x','y','z']],axis=1)
  y=np.loadtxt(io.BytesIO(read(f'/{split}/y_{split}.txt')),dtype=int)-1;g=np.loadtxt(io.BytesIO(read(f'/{split}/subject_{split}.txt')),dtype=int)
  # Keep alternate source rows per subject, without resetting at label transitions.
  idx=np.sort(np.concatenate([np.flatnonzero(g==u)[::2] for u in np.unique(g)]))
  for u in np.unique(g):assert np.all(np.diff(idx[g[idx]==u])>=2)
  xx.extend(X[idx]);yy.extend(y[idx]);gg.extend(g[idx]);ids.extend([f'{split}:{i}' for i in idx])
 save('har',xx,yy,gg,ids,{'sensor':'waist total acceleration xyz','duration_seconds':2.56,'windowing':'every other released row per subject, without resetting at label transitions; retained row indices are separated by at least two'})
if not (D/'dsads.npz').exists():
 z=zipfile.ZipFile(archive('dsads'));xx=[];yy=[];gg=[];ids=[]
 for name in sorted(n for n in z.namelist() if re.search(r'a\d+/p\d+/s\d+\.txt$',n)):
  a,p,_=map(int,re.search(r'a(\d+)/p(\d+)/s(\d+)\.txt$',name).groups());x=np.loadtxt(io.BytesIO(z.read(name)),delimiter=',',usecols=[0,1,2]);xx.append(resample(x));yy.append(a-1);gg.append(p);ids.append(name)
 save('dsads',xx,yy,gg,ids,{'sensor':'torso acceleration xyz','duration_seconds':5,'windowing':'released nonoverlapping segments; linear interpolation from 125 to 128 points'})
if not (D/'mhealth.npz').exists() or not (D/'mhealth_metadata.json').exists():
 z=zipfile.ZipFile(BASE/'raw'/'mhealth.zip');xx=[];yy=[];gg=[];ids=[];discard=0
 for name in sorted(n for n in z.namelist() if n.endswith('.log')):
  g=int(re.search(r'subject(\d+)',name).group(1));a=np.loadtxt(io.BytesIO(z.read(name)));y=a[:,-1].astype(int)
  borders=np.r_[0,np.flatnonzero(np.diff(y)!=0)+1,len(y)]
  for left,right in zip(borders[:-1],borders[1:]):
   if y[left]==0:discard+=right-left;continue
   for start in range(left,right-249,250):xx.append(resample(a[start:start+250,:3]));yy.append(y[start]-1);gg.append(g);ids.append(f'{name}:{start}:{start+250}')
 save('mhealth',xx,yy,gg,ids,{'sensor':'chest acceleration xyz','duration_seconds':5,'windowing':'nonoverlapping 250-point pure-label windows; linear interpolation to 128 points','null_samples_excluded':int(discard)})
if not (D/'wisdm.npz').exists():
 z=zipfile.ZipFile(BASE/'raw'/'wisdm.zip');z=zipfile.ZipFile(io.BytesIO(z.read('wisdm-dataset.zip')));xx=[];yy=[];gg=[];ids=[];qc=[];labels=list('ABCDEFGHIJKLMOPQRS')
 for name in sorted(n for n in z.namelist() if '/raw/phone/accel/' in n and n.endswith('.txt')):
  g=int(re.search(r'data_(\d+)_',name).group(1));raw=z.read(name).decode('utf-8').replace(';','');df=pd.read_csv(io.StringIO(raw),header=None,names=['g','label','time','x','y','z'],on_bad_lines='skip')
  for c in ['time','x','y','z']:df[c]=pd.to_numeric(df[c],errors='coerce')
  ok=np.isfinite(df[['time','x','y','z']]).all(axis=1)&df.label.isin(labels);bad=int((~ok).sum());df=df[ok]
  t=df.time.to_numpy(dtype='int64');x=df[['x','y','z']].to_numpy();y=df.label.to_numpy();dt=np.diff(t)
  borders=np.r_[0,np.flatnonzero((y[1:]!=y[:-1])|(dt<=0)|(dt>500_000_000))+1,len(t)];before=len(yy)
  for left,right in zip(borders[:-1],borders[1:]):
   if right-left<10:continue
   start=int(t[left]);end=int(t[right-1]);duration=5_000_000_000
   for tt in range(start,end-duration+1,duration):
    grid=tt+np.arange(128)*(duration/128);window=np.stack([np.interp(grid,t[left:right],x[left:right,j]) for j in range(3)])
    xx.append(window);yy.append(labels.index(y[left]));gg.append(g);ids.append(f'{name}:{tt}:{tt+duration}')
  qc.append(dict(subject=g,valid_rows=len(df),invalid_rows=bad,median_positive_dt_ns=float(np.median(dt[dt>0])),runs=len(borders)-1,windows=len(yy)-before));print('wisdm',qc[-1],flush=True)
 save('wisdm',xx,yy,gg,ids,{'sensor':'phone acceleration xyz','duration_seconds':5,'windowing':'timestamp-based nonoverlapping pure-label windows; split at nonincreasing timestamps or gaps >0.5 seconds; interpolate to 128 points','labels':labels,'per_subject_quality':qc})
