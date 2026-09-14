from pathlib import Path
import sys,itertools,json
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'revision5/fitdeps')]
import numpy as np
from horizon import crossing_probability,calibrated_brackets
records=[]
for N in [5,6,7]:
 for alpha in [.1,.3,.5]:
  k=int(np.ceil((N+1)*(1-alpha)-1e-12))
  times=tuple(range(1,N+1))
  for delta in [.05,.2]:
   bands,meta=calibrated_brackets(N,times,delta,alpha)
   failures=0;total=0
   for perm in itertools.permutations(range(1,N+1)):
    failed=False
    if k<=N:
     for t,(l,u) in zip(times,bands):
      j=sum(x<k for x in perm[:t]);r=k in perm[:t]
      failed|=(j+r<l or j>=u)
    failures+=failed;total+=1
   exact=failures/total
   assert exact<=delta+1e-12 and abs(exact-meta['crossing'])<1e-10
   records.append(dict(N=N,alpha=alpha,delta=delta,enumerated_crossing=exact,dp_crossing=meta['crossing'],permutations=total))
(HERE/'horizon_checks.json').write_text(json.dumps(dict(status='passed',cases=records),indent=2))
print('Exact DP agrees with exhaustive enumeration in all cases.',flush=True)
