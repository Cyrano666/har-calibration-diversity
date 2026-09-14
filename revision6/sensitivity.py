"""Prespecified exploratory sensitivities on the six seen datasets only."""
from pathlib import Path
import sys,time,json
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/('v2' if (HERE.parent/'v2').exists() else 'revision');sys.path[:0]=[str(HERE.parent/'revision5/fitdeps'),str(BASE)]
import numpy as np,pandas as pd
from sequential import trace,stop
from designs import scores,cutoff
rows=[]
for ds in ['har','dsads','mhealth','wisdm','pamap2','realdisp']:
 path=HERE.parent/'revision5/results/predictions'/f'{ds}_0_0_MR.npz'
 if not path.exists():path=BASE/'results/predictions'/path.name
 z=np.load(path);people=np.unique(z['cal_g']);ci={u:np.flatnonzero(z['cal_g']==u) for u in people};N=min(480,len(people)*120);times=tuple(sorted(set(list(range(20,N,20))+[N])))
 for kind in ['LAC','APS']:
  cm=scores(z['cal_p'],kind);truth=cm[np.arange(len(cm)),z['cal_y']];tm=scores(z['test_p'],kind)
  for draw in range(40):
   rng=np.random.default_rng(817233+draw);order=rng.permutation(people);quota=np.full(len(people),N//len(people));quota[:N%len(people)]+=1
   pool=rng.permutation(np.concatenate([rng.permutation(ci[u])[:n] for u,n in zip(order,quota)]));s=truth[pool];q=cutoff(s,.1)
   for delta in [.01,.05,.1,.2]:
    for mode in (['whole','even_to_odd'] if delta==.05 else ['whole']):
     monitor=tm if mode=='whole' else tm[::2];evaluation=tm if mode=='whole' else tm[1::2]
     r=stop(trace(s,monitor,times=times,engine='horizon',delta=delta),.5)
     inflation=((evaluation<=r['upper']).sum(1)-(evaluation<=q).sum(1)).mean()
     rows.append(dict(dataset=ds,score=kind,draw=draw,N=N,delta=delta,mode=mode,saving=1-r['t']/N,containment_failure=r['upper']<q,inflation=inflation,inflation_exceeded=inflation>.5+1e-12))
 print(ds,'complete',flush=True)
pd.DataFrame(rows).to_csv(HERE/'sensitivity.csv',index=False)
