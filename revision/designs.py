"""Cost-constrained design planning uses historical OOF subjects only."""
import numpy as np
def scores(p,kind):
 if kind=='LAC':return 1-p
 order=np.argsort(-p,axis=1,kind='stable');z=np.cumsum(np.take_along_axis(p,order,axis=1),axis=1);out=np.empty_like(z);np.put_along_axis(out,order,z,axis=1);return out
def cutoff(s,alpha):
 rank=int(np.ceil((len(s)+1)*(1-alpha)-1e-12))
 return np.inf if rank>len(s) else np.partition(s,rank-1)[rank-1]
def feasible(budget,cost,grid,nmax=120,nmin=10):
 return [(int(m),int(min(nmax,np.floor(budget/m-cost)))) for m in grid if np.floor(budget/m-cost)>=nmin]
class SubjectDistribution:
 def __init__(self,matrix,y):
  self.n=len(y);self.K=matrix.shape[1];self.true=np.sort(matrix[np.arange(len(y)),y]);self.all=np.sort(matrix.ravel())
  ordered=np.sort(matrix,axis=1);self.first=np.sort(ordered[:,0]);self.second=np.sort(ordered[:,1]);correct=np.argmin(matrix,axis=1)==y
  self.correct_first=np.sort(ordered[correct,0]);self.correct_second=np.sort(ordered[correct,1])
 def at(self,q):
  count=lambda a:np.searchsorted(a,q,side='right')
  single=count(self.first)-count(self.second);good=count(self.correct_first)-count(self.correct_second)
  return {'coverage':count(self.true)/self.n,'set_size':count(self.all)/self.n,'empty_rate':1-count(self.first)/self.n,'singleton_rate':single/self.n,'singleton_accuracy':good/single if single else np.nan}
class Planner:
 def __init__(self,p,y,g,kind,seed,draws=12):
  self.mat=scores(p,kind);self.y=y;self.g=g;self.people=np.unique(g);self.K=p.shape[1];self.draws=draws;self.cache={};self.records=[]
  self.idx={int(u):np.flatnonzero(g==u) for u in self.people};self.true={u:self.mat[ii,y[ii]] for u,ii in self.idx.items()};self.dist={u:SubjectDistribution(self.mat[ii],y[ii]) for u,ii in self.idx.items()}
  self.samples=[]
  for u in self.people:
   for draw in range(draws):
    rng=np.random.default_rng(seed+int(u)*10000+draw);pool=rng.permutation(self.people[self.people!=u]);perms={int(v):rng.permutation(len(self.idx[int(v)])) for v in pool};self.samples.append((int(u),pool,perms))
 def loss(self,m,n,alpha):
  key=(m,n,alpha)
  if key in self.cache:return self.cache[key]
  short=[];size=[]
  for u,pool,perms in self.samples:
   cal=np.concatenate([self.true[int(v)][perms[int(v)][:n]] for v in pool[:m]])
   assert len(cal)==m*n
   q=cutoff(cal,alpha);values=self.dist[u].at(q);short.append(max(0,1-alpha-values['coverage']));size.append(values['set_size']/self.K)
  value=(float(np.mean(short)),float(np.mean(size)));self.cache[key]=value;return value
 def variance_components(self,alpha):
  values=np.concatenate(list(self.true.values()));weights=np.concatenate([np.full(len(s),1/(len(self.people)*len(s))) for s in self.true.values()]);order=np.argsort(values);q=values[order][np.searchsorted(np.cumsum(weights[order]),1-alpha)]
  fu=np.array([np.mean(s<=q) for s in self.true.values()]);ns=np.array([len(s) for s in self.true.values()]);wu=fu*(1-fu)*ns/(ns-1)
  return max(0,float(fu.var(ddof=1)-np.mean(wu/ns))),float(wu.mean())
 def choose(self,budget,cost,grid,alpha=.1,nmax=120):
  candidates=feasible(budget,cost,grid,nmax);assert candidates
  # A common, deterministic tie rule prefers more independent subjects.
  aa,ww=self.variance_components(alpha);risk={z:self.loss(*z,alpha) for z in candidates}
  pick=lambda lam:min(candidates,key=lambda z:(risk[z][0]+lam*risk[z][1],-z[0]))
  fixed3=min(candidates,key=lambda z:(abs(z[0]-3),-z[0]))
  chosen={'one':candidates[0],'three':fixed3,'max':candidates[-1],'variance':min(candidates,key=lambda z:(aa/z[0]+ww/(z[0]*z[1]),-z[0])),'crossfit':pick(.05),'crossfit_l0':pick(0),'crossfit_l001':pick(.01),'crossfit_l01':pick(.1)}
  records=[dict(budget=budget,cost=cost,alpha=alpha,m=m,n=n,estimated_shortfall=risk[(m,n)][0],estimated_size_fraction=risk[(m,n)][1],variance_between=aa,variance_within=ww,variance_objective=aa/m+ww/(m*n)) for m,n in candidates]
  return chosen,records
