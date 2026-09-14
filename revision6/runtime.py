from pathlib import Path
import sys,time,json
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'revision5/fitdeps'))
import numpy as np,pandas as pd
from horizon import calibrated_brackets
from sequential import acquire
calibrated_brackets(20,(20,));rows=[];rng=np.random.default_rng(91871)
for N in [120,240,480,960]:
 times=tuple(range(20,N+1,20));s=rng.random(N);monitor=rng.random((1000,12))
 for rep in range(5):
  calibrated_brackets.cache_clear();t=time.perf_counter();bands,meta=calibrated_brackets(N,times,.05,.1);boundary=time.perf_counter()-t
  before=calibrated_brackets.cache_info();t=time.perf_counter();result=acquire(N,lambda a,b:s[a:b],monitor,tolerance=.5);online=time.perf_counter()-t
  after=calibrated_brackets.cache_info();assert after.hits==before.hits+1 and after.misses==before.misses
  rows.append(dict(N=N,rep=rep,boundary_seconds=boundary,online_seconds=online,queried=result['queried'],exact_crossing=meta['crossing'],includes_jit_compilation=False))
pd.DataFrame(rows).to_csv(HERE/'runtime.csv',index=False);print(pd.DataFrame(rows).groupby('N')[['boundary_seconds','online_seconds']].mean())
