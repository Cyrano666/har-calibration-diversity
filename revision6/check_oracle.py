from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'revision5/fitdeps'))
import numpy as np
from sequential import acquire,trace,stop
rng=np.random.default_rng(701834);records=[]
for N in [120,480]:
 s=rng.random(N);monitor=rng.random((71,12));times=tuple(range(20,N+1,20))
 for engine in ['horizon','hypergeom','martingale']:
  calls=[]
  def query(a,b):
   assert a==(calls[-1][1] if calls else 0)
   calls.append((a,b));return s[a:b]
  a=acquire(N,query,monitor,tolerance=1.,engine=engine)
  b=stop(trace(s,monitor,times=times,engine=engine),1.)
  assert a['queried']==b['t'] and a['upper']==b['upper'] and calls[-1][1]==a['queried']
  changed=s.copy();changed[a['queried']:]=100.
  c=stop(trace(changed,monitor,times=times,engine=engine),1.)
  assert c==b
  records.append(dict(N=N,engine=engine,queried=a['queried'],oracle_calls=calls))
(HERE/'oracle_checks.json').write_text(json.dumps(dict(status='passed',checks=records),indent=2))
print('Streaming oracle and unseen-label invariance checks passed.')
