"""Recompute exact crossing mass for every distinct main-study design and engine."""
from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'revision5/fitdeps'))
import numpy as np
from sequential import rank_brackets,martingale_brackets,reference_rank,checkpoints
from horizon import calibrated_brackets,crossing_probability
stamps=list((HERE/'evaluation').glob('*_audit.json'));assert len(stamps)==354
designs=set();implications=0
for path in stamps:
 record=json.loads(path.read_text());assert record['status']=='passed'
 implications+=record['numerical_implication_checks']
 for b in record['boundaries']:designs.add((b['N'],tuple(b['times'])))
results=[]
for N,times in sorted(designs):
 for engine in ['horizon','hypergeom','martingale','uncorrected','sparse_checks']:
  grid=checkpoints(N) if engine=='sparse_checks' else times
  if engine in ['horizon','sparse_checks']:ranks,_=calibrated_brackets(N,grid,.05,.1)
  elif engine=='martingale':ranks=martingale_brackets(N,grid,.05,.1)
  else:ranks=rank_brackets(N,grid,.05,.1,engine!='uncorrected')
  lower=np.zeros(N+1,dtype=np.int64);upper=np.arange(N+1,dtype=np.int64)+1
  for t,(l,u) in zip(grid,ranks):lower[t]=l;upper[t]=u
  failure=crossing_probability(N,reference_rank(N),lower,upper)
  if engine!='uncorrected':assert failure<=.05+1e-9,(N,engine,failure)
  results.append(dict(N=N,checkpoints=len(grid),engine=engine,exact_any_time_failure=float(failure)))
result=dict(status='passed',main_model_cases=354,unique_dense_designs=len(designs),numerical_stopping_implication_checks=implications,exact_boundary_checks=results,
            scope='Exact rank-path failure probability for each finite design, not population test coverage.')
(HERE/'design_audit.json').write_text(json.dumps(result,indent=2));print('Audited',len(results),'boundary engines/designs;',implications,'numerical stopping implications.')
