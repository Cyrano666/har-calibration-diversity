"""Evaluate the pre-specified HHAR acquisitions, preserving unfavorable results."""
from pathlib import Path
import sys,json,hashlib,argparse,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'revision'),str(ROOT/'revision6'),str(ROOT/'revision7')]
import numpy as np,pandas as pd
from designs import scores,cutoff,SubjectDistribution
from sequential import trace,stop
from prior_audit import select,bands
from horizon import calibrated_brackets
P=json.loads((HERE/'protocol.json').read_text());LOCK=json.loads((HERE/'protocol_lock.json').read_text())
for name,h in LOCK['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
OUT=HERE/'evaluation';OUT.mkdir(exist_ok=True)
def evaluate(path):
    target=OUT/(path.stem+'.csv.gz');audit=OUT/(path.stem+'_audit.json')
    if target.exists() and audit.exists():return
    z=np.load(path);_,rep,fold,model=path.stem.split('_');rep=int(rep);fold=int(fold)
    calpeople=np.unique(z['cal_g']);testpeople=np.unique(z['test_g']);assert len(calpeople)==2 and len(testpeople)==3
    assert not(set(calpeople)&set(testpeople) or set(z['train_g'])&set(calpeople) or set(z['train_g'])&set(testpeople))
    subjects=[];campaigns=[];implications=0
    for kind in ['LAC','APS']:
        cm=scores(z['cal_p'],kind);truth=cm[np.arange(len(cm)),z['cal_y']];tm=scores(z['test_p'],kind);flat=np.sort(tm.ravel());prepared=(flat,len(tm))
        ds={int(u):SubjectDistribution(tm[z['test_g']==u],z['test_y'][z['test_g']==u]) for u in testpeople}
        ci={int(u):np.flatnonzero(z['cal_g']==u) for u in calpeople}
        for N in [120,240]:
            times=tuple(range(20,N+1,20));accum={}
            for draw in range(40):
                rng=np.random.default_rng(P['query_seed_base']+rep*10000+fold*100+draw);order=rng.permutation(calpeople)
                pool=rng.permutation(np.concatenate([rng.permutation(ci[int(u)])[:N//2] for u in order]));s=truth[pool];q=cutoff(s,.1)
                ref=np.array([[d.at(q)['coverage'],d.at(q)['set_size']] for d in ds.values()])
                choices=[('full_reference',N,q,q)]
                for policy,engine in [('JRC','horizon'),('HG','hypergeom'),('CS_uniform','martingale')]:
                    r=stop(trace(s,tm,delta=.05,engine=engine,times=times,prepared_monitor=prepared),.5)
                    choices.append((policy,r['t'],r['lower'],r['upper']))
                t,lo,hi,_=select(s,flat,len(tm),9,1);choices.append(('CS_beta_9_1',t,lo,hi))
                for f in [.25,.5,.75]:
                    t=int(np.ceil(N*f));value=cutoff(s[:t],.1);choices.append((f'fixed_{int(f*100)}',t,value,value))
                for policy,t,lo,hi in choices:
                    vals=np.array([[d.at(hi)['coverage'],d.at(hi)['set_size']] for d in ds.values()])
                    inflation=(np.searchsorted(flat,hi,side='right')-np.searchsorted(flat,q,side='right'))/len(tm)
                    valid=lo<=q<=hi
                    if policy in ['JRC','HG','CS_uniform','CS_beta_9_1'] and valid:
                        assert inflation<=.5+1e-10 and np.all(vals[:,0]>=ref[:,0]-1e-12);implications+=1
                    record=np.column_stack([vals,vals-ref,np.full(3,t),np.full(3,1-t/N)])
                    accum[policy]=accum.get(policy,np.zeros_like(record))+record/40
                    campaigns.append(dict(model=model,rep=rep,fold=fold,score=kind,N=N,draw=draw,policy=policy,queried=t,saving=1-t/N,bracket_failure=not valid,threshold_failure=hi<q,inflation_failure=inflation>.5+1e-10,actual_inflation=inflation))
            for policy,vals in accum.items():
                for user,v in zip(testpeople,vals):subjects.append(dict(model=model,rep=rep,fold=fold,subject=int(user),score=kind,N=N,policy=policy,coverage=v[0],set_size=v[1],delta_coverage=v[2],delta_size=v[3],queried=v[4],saving=v[5]))
    pd.DataFrame(subjects).to_csv(target,index=False,compression='gzip')
    pd.DataFrame(campaigns).to_csv(OUT/(path.stem+'_campaign.csv.gz'),index=False,compression='gzip')
    audit.write_text(json.dumps(dict(status='passed',prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),protocol_sha256=LOCK['files']['confirmation_hhar/protocol.json'],implication_checks=implications,subject_rows=len(subjects),campaign_rows=len(campaigns)),indent=2))
    print(path.stem,'evaluated',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    while True:
        files=sorted(p for p in (HERE/'results/predictions').glob('*.npz') if p.with_suffix('.json').exists())
        for path in files:evaluate(path)
        if len(files)==45:print('All 45 HHAR acquisition cases complete',flush=True);break
        if not args.watch:print(len(files),'available model cases evaluated; full study incomplete');break
        time.sleep(3)
