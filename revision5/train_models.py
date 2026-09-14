from pathlib import Path
import sys,os,json,time,argparse,hashlib,warnings,shutil,gc
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/('v2' if (HERE.parent/'v2').exists() else 'revision');sys.path.insert(0,str(HERE/'fitdeps'))
from models import create_model
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from threadpoolctl import threadpool_limits
P=json.loads((HERE/'final_protocol.json').read_text());LOCK=json.loads((HERE/'protocol_lock.json').read_text());assert hashlib.sha256((HERE/'final_protocol.json').read_bytes()).hexdigest()==LOCK['protocol_sha256']
OLD=json.loads((BASE/'protocol_v2.json').read_text());OUT=HERE/'results/predictions';OUT.mkdir(parents=True,exist_ok=True)

def splits(ds,people):
 cfg=P[ds] if ds in P['confirmation_datasets'] else OLD[ds];seeds=P['confirmation_seeds'] if ds in P['confirmation_datasets'] else P['development_seeds']
 for rep,seed in enumerate(seeds):
  order=np.random.default_rng(seed).permutation(people)
  for fold,test in enumerate(np.array_split(order,cfg['folds'])):
   rest=np.random.default_rng(seed+101*fold).permutation(np.setdiff1d(people,test));tr=rest[:cfg['train_subjects']];ca=rest[cfg['train_subjects']:]
   assert not(set(tr)&set(ca) or set(tr)&set(test) or set(ca)&set(test)) and len(ca)>cfg['max_cal_subjects']
   yield dict(dataset=ds,rep=rep,fold=fold,seed=seed,train_subjects=tr.tolist(),cal_subjects=ca.tolist(),test_subjects=test.tolist())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='+',default=P['confirmation_datasets']);ap.add_argument('--models',nargs='+',default=P['models']);ap.add_argument('--limit',type=int);args=ap.parse_args()
 for ds in args.datasets:
  file=(HERE/'data' if ds in P['confirmation_datasets'] else BASE/'data')/(ds+'.npz');data=np.load(file);X,Z,y,g=[data[k] for k in ['X','Z','y','g']];K=len(np.unique(y));specs=list(splits(ds,np.unique(g)))
  if args.limit:specs=specs[:args.limit]
  for sp in specs:
   tr=np.flatnonzero(np.isin(g,sp['train_subjects']));ca=np.flatnonzero(np.isin(g,sp['cal_subjects']));te=np.flatnonzero(np.isin(g,sp['test_subjects']))
   for name in args.models:
    stem=f'{ds}_{sp["rep"]}_{sp["fold"]}_{name}';path=OUT/(stem+'.npz')
    if path.exists() and path.with_suffix('.json').exists():continue
    original=BASE/'results/predictions'/(stem+'.npz')
    if ds in P['development_datasets'] and name in ['LR','MR'] and original.exists():
     shutil.copy2(original,path);shutil.copy2(original.with_suffix('.json'),path.with_suffix('.json'));continue
    start=time.perf_counter();seed=sp['seed']+sp['fold'];innerlog=[]
    with threadpool_limits(4),warnings.catch_warnings(record=True) as warn:
     warnings.simplefilter('always');m=create_model(name,seed).fit(X[tr],Z[tr],y[tr]);pc=np.zeros((len(ca),K));pt=np.zeros((len(te),K));pc[:,m.classes_]=m.proba(X[ca],Z[ca]);pt[:,m.classes_]=m.proba(X[te],Z[te]);main_time=time.perf_counter()-start
     poof=np.zeros((len(tr),K));groups=np.array_split(np.random.default_rng(seed+444).permutation(np.unique(g[tr])),3)
     for j,held in enumerate(groups):
      ii=np.flatnonzero(~np.isin(g[tr],held));oo=np.flatnonzero(np.isin(g[tr],held));assert not set(g[tr[ii]])&set(g[tr[oo]])
      aux=create_model(name,seed+1000+j).fit(X[tr[ii]],Z[tr[ii]],y[tr[ii]]);poof[np.ix_(oo,aux.classes_)]=aux.proba(X[tr[oo]],Z[tr[oo]]);innerlog.append(dict(train_subjects=np.unique(g[tr[ii]]).tolist(),validation_subjects=held.tolist(),classes_seen=aux.classes_.tolist()));del aux;gc.collect()
    for a in [pc,pt,poof]:assert np.isfinite(a).all() and np.allclose(a.sum(1),1,atol=1e-6)
    np.savez_compressed(path,cal_p=pc,test_p=pt,train_oof_p=poof,cal_y=y[ca],test_y=y[te],train_y=y[tr],cal_g=g[ca],test_g=g[te],train_g=g[tr],cal_index=ca,test_index=te,train_index=tr)
    record=dict(**sp,model=name,total_fit_seconds=time.perf_counter()-start,main_fit_seconds=main_time,inner_splits=innerlog,warnings=[str(w.message) for w in warn],protocol_sha256=LOCK['protocol_sha256'],main_classes_seen=m.classes_.tolist(),test_metrics=[dict(subject=int(u),n=int(np.sum(g[te]==u)),accuracy=accuracy_score(y[te][g[te]==u],pt[g[te]==u].argmax(1)),macro_f1=f1_score(y[te][g[te]==u],pt[g[te]==u].argmax(1),labels=np.arange(K),average='macro',zero_division=0)) for u in np.unique(g[te])])
    if name=='IT':record.update(parameters=m.parameters,epoch_losses=m.epoch_losses)
    path.with_suffix('.json').write_text(json.dumps(record,indent=2));print(stem,round(record['total_fit_seconds'],1),'seconds',len(record['warnings']),'warnings',flush=True);del m;gc.collect()
if __name__=='__main__':main()
