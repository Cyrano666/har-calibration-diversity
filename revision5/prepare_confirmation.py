"""Prepare reserved benchmarks using schema/eligibility only; no predictive evaluation."""
from pathlib import Path
import sys,io,zipfile,re,json,hashlib
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'fitdeps'))
import numpy as np
from scipy.signal import resample_poly
OUT=HERE/'data';OUT.mkdir(exist_ok=True)

def statistics(x):
 a=np.concatenate([x,np.sqrt((x*x).sum(axis=1))[:,None,:]],axis=1)
 feats=[a.mean(2),a.std(2),a.min(2),a.max(2),np.sqrt((a*a).mean(2)),np.abs(np.diff(a,axis=2)).mean(2),np.quantile(a,.25,axis=2),np.quantile(a,.75,axis=2)]
 for i,j in [(0,1),(0,2),(1,2)]:
  u=x[:,i]-x[:,i].mean(1)[:,None];v=x[:,j]-x[:,j].mean(1)[:,None];den=np.sqrt((u*u).sum(1)*(v*v).sum(1));feats.append(np.divide((u*v).sum(1),den,out=np.zeros(len(x)),where=den>1e-12)[:,None])
 return np.concatenate(feats,axis=1).astype('float32')

def prepare(ds):
 if (OUT/(ds+'.npz')).exists():return
 x=[];y=[];g=[];ids=[];qc=[]
 with zipfile.ZipFile(HERE/'raw'/(ds+'.zip')) as z:
  files=sorted(n for n in z.namelist() if ('/Protocol/' in n and n.endswith('.dat')) if ds=='pamap2') if ds=='pamap2' else sorted(n for n in z.namelist() if n.endswith('_ideal.log'))
  for name in files:
   raw=z.read(name);user=int(re.search(r'subject(\d+)',name)[1]);before=len(x);rejected=0
   if ds=='pamap2':
    a=np.loadtxt(io.BytesIO(raw),usecols=[0,1,21,22,23]);t=a[:,0];label=a[:,1].astype(int);acc=a[:,2:];samples=500;up=32;down=125;gap=.05
   else:
    a=np.loadtxt(io.BytesIO(raw),usecols=[0,1,28,29,30,119]);t=a[:,0]+a[:,1]*1e-6;label=a[:,-1].astype(int);acc=a[:,2:5];samples=250;up=64;down=125;gap=.06
   dt=np.diff(t);borders=np.r_[0,np.flatnonzero((label[1:]!=label[:-1])|(dt<=0)|(dt>gap))+1,len(t)]
   for left,right in zip(borders[:-1],borders[1:]):
    if label[left]<=0:continue
    for start in range(left,right-samples+1,samples):
     window=acc[start:start+samples].copy();finite=np.isfinite(window)
     if np.any(finite.mean(axis=0)<.98):rejected+=1;continue
     for j in range(3):
      if not finite[:,j].all():window[:,j]=np.interp(np.arange(samples),np.flatnonzero(finite[:,j]),window[finite[:,j],j])
     value=resample_poly(window,up,down,axis=0,padtype='line').T.astype('float32');assert value.shape==(3,128)
     x.append(value);y.append(int(label[start]));g.append(user);ids.append(f'{name}:{start}:{start+samples}')
   qc.append(dict(file=name,subject=user,raw_rows=len(t),retained_windows=len(x)-before,rejected_missing_windows=rejected,sha256=hashlib.sha256(raw).hexdigest()))
   print(ds,user,len(x)-before,'windows',flush=True)
 X=np.asarray(x,dtype='float32');y=np.asarray(y);g=np.asarray(g);ids=np.asarray(ids)
 counts={int(u):int(np.sum(g==u)) for u in np.unique(g)};eligible=[u for u,n in counts.items() if n>=120];keep=np.isin(g,eligible)
 X=X[keep];y=y[keep];g=g[keep];ids=ids[keep];classes=np.unique(y);mapping={int(c):i for i,c in enumerate(classes)};y=np.array([mapping[int(v)] for v in y],dtype='int32');Z=statistics(X)
 assert np.isfinite(X).all() and np.isfinite(Z).all()
 np.savez_compressed(OUT/(ds+'.npz'),X=X,Z=Z,y=y,g=g.astype('int32'),ids=ids)
 meta=dict(dataset=ds,windows=len(y),subjects=len(eligible),classes=len(classes),original_labels=classes.tolist(),subject_counts=counts,eligible_subjects=eligible,excluded_for_less_than_120_windows=[u for u in counts if u not in eligible],sensor='chest ±16g acceleration' if ds=='pamap2' else 'BACK acceleration (third sensor in original manual)',zero_based_signal_columns=[21,22,23] if ds=='pamap2' else [28,29,30],duration_seconds=5,shape=list(X.shape),feature_count=35,resampling='SciPy resample_poly, anti-aliased to 128 points with line padding',missing_policy='reject if any axis has >2% nonfinite points in a window; otherwise interpolate within window',quality=qc,signal_sha256=hashlib.sha256(X.tobytes()).hexdigest(),stage='Metadata/preprocessing only, before final method lock; no predictions inspected')
 (OUT/(ds+'_metadata.json')).write_text(json.dumps(meta,indent=2));print(ds,len(y),'windows',len(eligible),'subjects',len(classes),'classes',flush=True)

if __name__=='__main__':
 for ds in ['pamap2','realdisp']:prepare(ds)
