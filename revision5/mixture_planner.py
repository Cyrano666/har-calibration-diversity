"""Sparse randomized acquisition plans: LP optimality does not imply target validity."""
import numpy as np
from scipy.optimize import linprog

def plan_mixtures(records,epsilon=.01):
 z=[(r['m'],r['n'],r['blocks']) for r in records];cost=np.array([r['spent'] for r in records],float)
 mean=np.array([r['delta'] for r in records]);se=np.array([r['user_se'] for r in records]);ref=next(i for i,r in enumerate(records) if r['reference'])
 result={};audits={}
 for name,kappa,restriction in [('sparse_mixture',.5,None),('mixture_mean',0,None),('mixture_guard1',1,None),('mixture_fixed4',.5,z[ref][2]),('mixture_fixed8',.5,8)]:
  keep=np.array([i for i in range(len(z)) if restriction is None or z[i][2]==restriction or i==ref])
  A=(mean+kappa*se).T[:,keep]
  lp=linprog(cost[keep],A_ub=A,b_ub=np.full(3,epsilon),A_eq=np.ones((1,len(keep))),b_eq=[1],bounds=(0,None),method='highs-ds',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
  assert lp.success,lp.message
  weights=np.zeros(len(z));weights[keep]=lp.x;weights[weights<1e-10]=0;weights/=weights.sum()
  assert np.all((mean+kappa*se).T@weights<=epsilon+1e-7)
  assert weights@cost<=cost[ref]+1e-7
  support=np.flatnonzero(weights>0);assert len(support)<=4
  result[name]=[(z[i],float(weights[i])) for i in support]
  audits[name]=dict(expected_cost=float(weights@cost),support=len(support),source_guard=((mean+kappa*se).T@weights).tolist(),solver_status=lp.message)
 # Strong simple control: best mixture of one candidate with the reference.
 A=mean+.5*se;best=(cost[ref],ref,0.)
 for i in range(len(z)):
  positive=A[i]>epsilon
  w=min(1.,float(np.min(epsilon/A[i][positive]))) if positive.any() else 1.
  value=w*cost[i]+(1-w)*cost[ref]
  if value<best[0]-1e-9:best=(value,i,w)
 _,i,w=best;result['two_plan']=[(z[i],w),(z[ref],1-w)] if w>0 else [(z[ref],1.)]
 return result,audits
