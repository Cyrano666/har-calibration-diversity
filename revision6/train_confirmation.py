from pathlib import Path
import sys,os,json,hashlib,time,warnings,gc
HERE=Path(__file__).resolve().parent;V5=HERE.parent/'revision5';sys.path[:0]=[str(V5/'fitdeps'),str(V5)]
from models import create_model
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from threadpoolctl import threadpool_limits
P=json.loads((HERE/'protocol.json').read_text());LOCK=json.loads((HERE/'protocol_lock.json').read_text())
for f,digest in LOCK['files'].items():assert hashlib.sha256((HERE/f).read_bytes()).hexdigest()==digest
z=np.load(HERE/'data/uschad.npz');X,Z,y,g=[z[k] for k in ['X','Z','y','g']];people=np.unique(g);K=len(np.unique(y))
OUT=HERE/'results/predictions';OUT.mkdir(parents=True,exist_ok=True)
for rep,seed in enumerate(P['confirmation_seeds']):
 order=np.random.default_rng(seed).permutation(people)
 for fold,tepeople in enumerate(np.array_split(order,7)):
  rest=np.random.default_rng(seed+101*fold).permutation(np.setdiff1d(people,tepeople));trpeople=rest[:8];capeople=rest[8:]
  tr=np.flatnonzero(np.isin(g,trpeople));ca=np.flatnonzero(np.isin(g,capeople));te=np.flatnonzero(np.isin(g,tepeople))
  assert len(trpeople)==8 and len(capeople)==4 and len(tepeople)==2 and len(np.unique(np.r_[tr,ca,te]))==len(y)
  for model in P['models']:
   stem=f'uschad_{rep}_{fold}_{model}';path=OUT/(stem+'.npz')
   if path.exists() and path.with_suffix('.json').exists():continue
   start=time.perf_counter()
   with threadpool_limits(4),warnings.catch_warnings(record=True) as warn:
    warnings.simplefilter('always');m=create_model(model,seed+fold).fit(X[tr],Z[tr],y[tr])
    pc=np.zeros((len(ca),K));pt=np.zeros((len(te),K));pc[:,m.classes_]=m.proba(X[ca],Z[ca]);pt[:,m.classes_]=m.proba(X[te],Z[te])
   for a in [pc,pt]:assert np.isfinite(a).all() and np.allclose(a.sum(1),1,atol=1e-6)
   np.savez_compressed(path,cal_p=pc,test_p=pt,cal_y=y[ca],test_y=y[te],cal_g=g[ca],test_g=g[te],train_g=g[tr],cal_index=ca,test_index=te,train_index=tr)
   record=dict(dataset='uschad',rep=rep,fold=fold,seed=seed,model=model,train_subjects=trpeople.tolist(),cal_subjects=capeople.tolist(),test_subjects=tepeople.tolist(),
    fit_seconds=time.perf_counter()-start,warnings=[str(w.message) for w in warn],protocol_sha256=LOCK['files']['protocol.json'],
    test_metrics=[dict(subject=int(u),n=int(np.sum(g[te]==u)),accuracy=accuracy_score(y[te][g[te]==u],pt[g[te]==u].argmax(1)),macro_f1=f1_score(y[te][g[te]==u],pt[g[te]==u].argmax(1),labels=np.arange(K),average='macro',zero_division=0)) for u in tepeople])
   if model=='IT':record.update(parameters=m.parameters,epoch_losses=m.epoch_losses)
   path.with_suffix('.json').write_text(json.dumps(record,indent=2));print(stem,round(record['fit_seconds'],1),'seconds',len(warn),'warnings',flush=True)
   del m;gc.collect()
print('All 105 confirmation model fits complete.',flush=True)
