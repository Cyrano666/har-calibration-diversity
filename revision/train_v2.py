"""Four-dataset, grouped outer evaluation with historical cross-fitted predictions."""
from pathlib import Path
import sys,os,time,json,warnings,hashlib,importlib.metadata,argparse
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'pydeps'))
os.environ['NUMBA_NUM_THREADS']='4';os.environ['NUMBA_CACHE_DIR']=str(BASE/'numba_cache');os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score,accuracy_score
from threadpoolctl import threadpool_limits
P=json.loads((BASE/'protocol_v2.json').read_text());O=BASE/'results';O.mkdir(exist_ok=True);(O/'predictions').mkdir(exist_ok=True)
class Model:
 def __init__(self,name,seed):self.name=name;self.seed=seed
 def fit(self,X,Z,y):
  if self.name=='MR':
   from aeon.transformations.collection.convolution_based import MiniRocket
   self.mean=X.mean(axis=(0,2),keepdims=True);self.std=np.maximum(X.std(axis=(0,2),keepdims=True),1e-6)
   self.transform=MiniRocket(n_kernels=P['model_parameters']['minirocket_kernels'],n_jobs=4,random_state=self.seed)
   z=self.transform.fit_transform(((X-self.mean)/self.std).astype('float32'))
  else:z=Z
  if self.name=='ET':self.model=ExtraTreesClassifier(n_estimators=300,max_features='sqrt',n_jobs=4,random_state=self.seed)
  else:
   self.scaler=StandardScaler();z=self.scaler.fit_transform(z)
   self.model=LogisticRegression(C=1,max_iter=2000,tol=1e-4,solver='lbfgs',random_state=self.seed)
  self.model.fit(z,y);self.classes_=self.model.classes_;return self
 def proba(self,X,Z):
  z=self.transform.transform(((X-self.mean)/self.std).astype('float32')) if self.name=='MR' else Z
  if self.name!='ET':z=self.scaler.transform(z)
  return self.model.predict_proba(z)
def specs(ds,subjects):
 out=[]
 for rep,seed in enumerate(P['seeds']):
  rng=np.random.default_rng(seed);order=rng.permutation(subjects)
  for fold,tests in enumerate(np.array_split(order,P[ds]['folds'])):
   rest=np.random.default_rng(seed+101*fold).permutation(np.setdiff1d(subjects,tests));train=rest[:P[ds]['train_subjects']];cal=rest[P[ds]['train_subjects']:]
   assert len(cal)>P[ds]['max_cal_subjects']
   out.append(dict(dataset=ds,rep=rep,fold=fold,seed=seed,train_subjects=train.tolist(),cal_subjects=cal.tolist(),test_subjects=tests.tolist()))
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='+',default=P['datasets']);ap.add_argument('--models',nargs='+',default=P['models']);ap.add_argument('--limit',type=int);args=ap.parse_args()
 allsplits=[];rows=[];records=[]
 for ds in P['datasets']:
  d=np.load(BASE/'data'/(ds+'.npz'));allsplits.extend(specs(ds,np.unique(d['g'])))
 (O/'splits.json').write_text(json.dumps(allsplits,indent=2))
 for ds in args.datasets:
  d=np.load(BASE/'data'/(ds+'.npz'));X=d['X'];Z=d['Z'];y=d['y'];g=d['g'];K=len(np.unique(y));ss=[s for s in allsplits if s['dataset']==ds]
  if args.limit:ss=ss[:args.limit]
  for sp in ss:
   tr=np.flatnonzero(np.isin(g,sp['train_subjects']));ca=np.flatnonzero(np.isin(g,sp['cal_subjects']));te=np.flatnonzero(np.isin(g,sp['test_subjects']))
   assert not(set(g[tr])&set(g[ca]) or set(g[tr])&set(g[te]) or set(g[ca])&set(g[te]))
   for name in args.models:
    stem=f'{ds}_{sp["rep"]}_{sp["fold"]}_{name}';path=O/'predictions'/(stem+'.npz')
    if path.exists() and path.with_suffix('.json').exists():continue
    start=time.perf_counter();seed=sp['seed']+sp['fold'];logs=[]
    with threadpool_limits(limits=4),warnings.catch_warnings(record=True) as ww:
     warnings.simplefilter('always');m=Model(name,seed).fit(X[tr],Z[tr],y[tr]);pt=m.proba(X[te],Z[te]);pc=m.proba(X[ca],Z[ca]);fulltime=time.perf_counter()-start
     poof=np.zeros((len(tr),K));innergroups=np.array_split(np.random.default_rng(seed+444).permutation(np.unique(g[tr])),3)
     for inner,held in enumerate(innergroups):
      ii=np.flatnonzero(~np.isin(g[tr],held));oo=np.flatnonzero(np.isin(g[tr],held));assert not set(g[tr][ii])&set(g[tr][oo])
      aux=Model(name,seed+1000+inner).fit(X[tr[ii]],Z[tr[ii]],y[tr[ii]])
      poof[np.ix_(oo,aux.classes_)]=aux.proba(X[tr[oo]],Z[tr[oo]])
      logs.append({'train_subjects':np.unique(g[tr[ii]]).tolist(),'validation_subjects':held.tolist(),'classes_seen':aux.classes_.tolist()})
     warn=[str(w.message) for w in ww]
    assert np.isfinite(poof).all() and np.allclose(poof.sum(1),1) and np.allclose(pc.sum(1),1) and np.allclose(pt.sum(1),1)
    info=dict(**sp,model=name,total_fit_seconds=time.perf_counter()-start,main_fit_seconds=fulltime,inner_splits=logs,warnings=warn)
    np.savez_compressed(path,cal_p=pc,test_p=pt,train_oof_p=poof,cal_y=y[ca],test_y=y[te],train_y=y[tr],cal_g=g[ca],test_g=g[te],train_g=g[tr],cal_index=ca,test_index=te,train_index=tr)
    info['test_metrics']=[dict(subject=int(u),n=int(np.sum(g[te]==u)),accuracy=accuracy_score(y[te][g[te]==u],pt[g[te]==u].argmax(1)),macro_f1=f1_score(y[te][g[te]==u],pt[g[te]==u].argmax(1),labels=np.arange(K),average='macro',zero_division=0)) for u in np.unique(g[te])]
    path.with_suffix('.json').write_text(json.dumps(info,indent=2));print(stem,round(info['total_fit_seconds'],1),'seconds',len(warn),'warnings',flush=True)
 (O/'environment.json').write_text(json.dumps({'versions':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy','scikit-learn','aeon','numba','matplotlib']},'python':sys.version,'protocol_sha256':hashlib.sha256((BASE/'protocol_v2.json').read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
