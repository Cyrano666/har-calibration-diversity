"""Locked reference-emulation evaluation. Original labels are evaluation oracles."""
from pathlib import Path
import sys,json,hashlib,time,argparse
HERE=Path(__file__).resolve().parent;V5=HERE.parent/'revision5';OLD=HERE.parent/('v2' if (HERE.parent/'v2').exists() else 'revision')
sys.path[:0]=[str(V5/'fitdeps'),str(OLD)]
import numpy as np,pandas as pd
from designs import scores,cutoff,SubjectDistribution
from sequential import trace,stop,checkpoints
from horizon import calibrated_brackets
P=json.loads((HERE/'protocol.json').read_text());LOCK=json.loads((HERE/'protocol_lock.json').read_text())
for f,digest in LOCK['files'].items():assert hashlib.sha256((HERE/f).read_bytes()).hexdigest()==digest
OUT=HERE/'evaluation';OUT.mkdir(exist_ok=True)

def evaluate(path):
 dest=OUT/(path.stem+'.csv.gz');camp=OUT/(path.stem+'_campaign.csv.gz');audit=OUT/(path.stem+'_audit.json')
 if dest.exists() and camp.exists() and audit.exists():return
 ds,rep,fold,model=path.stem.split('_');rep=int(rep);fold=int(fold)
 z=np.load(path);people=np.unique(z['cal_g']);testpeople=np.unique(z['test_g'])
 assert not(set(z['cal_g'])&set(z['test_g']) or set(z['train_g'])&set(z['cal_g']) or set(z['train_g'])&set(z['test_g']))
 ci={int(u):np.flatnonzero(z['cal_g']==u) for u in people};assert min(map(len,ci.values()))>=120
 results=[];campaigns=[];boundary_audit=[];start=time.perf_counter();checks=0
 for kind in ['LAC','APS']:
  cm=scores(z['cal_p'],kind);truth=cm[np.arange(len(cm)),z['cal_y']];tm=scores(z['test_p'],kind);K=tm.shape[1]
  prepared=(np.sort(tm.ravel()),len(tm));distributions={int(u):SubjectDistribution(tm[z['test_g']==u],z['test_y'][z['test_g']==u]) for u in testpeople}
  for ceiling in P['pool_size_ceilings']:
   N=min(ceiling,120*len(people));times=tuple(sorted(set(list(range(20,N,20))+[N])))
   bands,meta=calibrated_brackets(N,times,P['delta'],P['alpha']);boundary_audit.append(dict(score=kind,N=N,times=times,bands=bands,**meta));accum={}
   for draw in range(P['calibration_draws']):
    rng=np.random.default_rng(P['query_seed_base']+rep*10000+fold*100+draw)
    order=rng.permutation(people);quotas=np.full(len(order),N//len(order));quotas[:N%len(order)]+=1
    pool=np.concatenate([rng.permutation(ci[int(u)])[:n] for u,n in zip(order,quotas)]);pool=rng.permutation(pool)
    assert len(pool)==N and len(np.unique(pool))==N
    s=truth[pool];qref=cutoff(s,.1);refvalues=np.array([[d.at(qref)['coverage'],d.at(qref)['set_size'],max(0,.9-d.at(qref)['coverage'])] for d in distributions.values()])
    choices=[('full_reference',.5,N,qref,qref,0.)]
    for engine in ['horizon','hypergeom','martingale','uncorrected','no_intersection','sparse_checks']:
     records=trace(s,tm,delta=.05,engine='horizon' if engine in ['no_intersection','sparse_checks'] else engine,
                   spending=engine!='uncorrected',nested=engine!='no_intersection',
                   times=checkpoints(N) if engine=='sparse_checks' else times,prepared_monitor=prepared)
     tolerances=[.1,.25,.5,1.] if engine in ['horizon','hypergeom','martingale'] else [.5]
     for tolerance in tolerances:
      r=stop(records,tolerance);choices.append((engine,tolerance,r['t'],r['lower'],r['upper'],r['set_width']))
      if engine=='horizon' and tolerance==.5:
       midpoint=(r['lower']+r['upper'])/2
       if not np.isfinite(midpoint):midpoint=cutoff(s[:r['t']],.1)
       choices.append(('midpoint',.5,r['t'],r['lower'],midpoint,r['set_width']))
    for fraction in [.25,.5,.75]:
     t=max(1,int(np.ceil(N*fraction)));q=cutoff(s[:t],.1);choices.append((f'fixed_{int(fraction*100)}',.5,t,q,q,np.nan))
    t=int(np.ceil(N*.5));r=trace(s,tm,times=(t,N),prepared_monitor=prepared)[0]
    choices.append(('one_look_50',.5,t,r['lower'],r['upper'],r['set_width']))
    for name,tolerance,t,lo,hi,width in choices:
     values=np.array([[d.at(hi)['coverage'],d.at(hi)['set_size'],max(0,.9-d.at(hi)['coverage'])] for d in distributions.values()])
     inflation=(np.searchsorted(prepared[0],hi,side='right')-np.searchsorted(prepared[0],qref,side='right'))/len(tm)
     valid=lo<=qref<=hi
     if name in ['horizon','hypergeom','martingale','no_intersection','sparse_checks'] and valid:
      assert inflation<=tolerance+1e-10 and np.all(values[:,0]>=refvalues[:,0]-1e-12)
      checks+=1
     key=(name,tolerance)
     record=np.column_stack([values,values-refvalues,np.full(len(values),t),np.full(len(values),1-t/N)])
     accum[key]=accum.get(key,np.zeros_like(record))+record/P['calibration_draws']
     campaigns.append(dict(dataset=ds,model=model,rep=rep,fold=fold,score=kind,ceiling=ceiling,N=N,draw=draw,policy=name,tolerance=tolerance,
       queried=t,saving=1-t/N,bracket_failure=not valid,containment_failure=hi<qref,inflation_exceeded=inflation>tolerance+1e-10,
       actual_inflation=inflation,reference_q=qref,lower=lo,upper=hi,certified_width=width))
   for (name,tolerance),values in accum.items():
    for u,v in zip(testpeople,values):
     results.append(dict(dataset=ds,model=model,rep=rep,fold=fold,subject=int(u),score=kind,ceiling=ceiling,N=N,participants=len(people),K=K,policy=name,tolerance=tolerance,
       coverage=v[0],set_size=v[1],shortfall=v[2],delta_coverage=v[3],delta_size=v[4],delta_shortfall=v[5],queried=v[6],saving=v[7]))
  print(path.stem,kind,round(time.perf_counter()-start,2),'s',flush=True)
 pd.DataFrame(results).to_csv(dest,index=False,compression='gzip');pd.DataFrame(campaigns).to_csv(camp,index=False,compression='gzip')
 audit.write_text(json.dumps(dict(status='passed',predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),protocol_sha256=LOCK['files']['protocol.json'],
   numerical_implication_checks=checks,subject_rows=len(results),campaign_rows=len(campaigns),seconds=time.perf_counter()-start,boundaries=boundary_audit),indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='+',default=P['development_datasets']);ap.add_argument('--models',nargs='+',default=P['models']);ap.add_argument('--limit',type=int);ap.add_argument('--output',type=Path);args=ap.parse_args()
 if args.output:OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
 paths={}
 for folder in [OLD/'results/predictions',V5/'results/predictions',HERE/'results/predictions']:
  for path in folder.glob('*.npz'):
   if path.stem.split('_')[0] in args.datasets and path.stem.split('_')[-1] in args.models and path.with_suffix('.json').exists():paths[path.stem]=path
 ordered=sorted(paths.values(),key=lambda x:x.stem)
 if args.limit:ordered=ordered[:args.limit]
 for p in ordered:evaluate(p)
