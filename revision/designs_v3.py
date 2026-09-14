import numpy as np
from designs import Planner,feasible,cutoff
class CostPlanner(Planner):
 def summary(self,m,n,alpha):
  if not hasattr(self,'extended_cache'):self.extended_cache={}
  key=(m,n,alpha)
  if key in self.extended_cache:return self.extended_cache[key]
  cov=[];gap=[];size=[]
  for u,pool,perms in self.samples:
   sample=np.concatenate([self.true[int(v)][perms[int(v)][:n]] for v in pool[:m]])
   assert len(sample)==m*n
   val=self.dist[u].at(cutoff(sample,alpha));cov.append(val['coverage']);gap.append(max(0,1-alpha-val['coverage']));size.append(val['set_size']/self.K)
  ans=(float(np.mean(cov)),float(np.mean(gap)),float(np.mean(size)));self.extended_cache[key]=ans;return ans
 def choose_cost(self,B,c,grid,alpha=.1):
  original=feasible(B,c,grid);ref=original[-1]
  candidates=sorted(set([(m,n) for m in grid for n in [10,20,40,80,120] if m*(c+n)<=B]+[ref]))
  values={z:self.summary(*z,alpha) for z in candidates};r=values[ref];a,w=self.variance_components(alpha);vr=a/ref[0]+w/(ref[0]*ref[1])
  bycost=lambda z:(z[0]*(c+z[1]),-z[0],-z[1])
  def pick(eps):
   keep=[z for z in candidates if values[z][0]>=r[0]-eps-1e-12 and values[z][1]<=r[1]+eps+1e-12 and values[z][2]<=r[2]+eps+1e-12]
   assert ref in keep
   return min(keep,key=bycost)
  vkeep=[z for z in candidates if a/z[0]+w/(z[0]*z[1])<=1.1*vr+1e-12];assert ref in vkeep
  initial=min(original,key=lambda z:(self.summary(*z,alpha)[1]+.05*self.summary(*z,alpha)[2],-z[0]))
  policies={'full':ref,'half':(ref[0],max(10,ref[1]//2)),'variance_save':min(vkeep,key=bycost),'initial_CFDS':initial,'CFCR':pick(.01),'CFCR_0':pick(0),'CFCR_005':pick(.005),'CFCR_02':pick(.02)}
  records=[dict(m=m,n=n,spent=m*(c+n),coverage=v[0],shortfall=v[1],size_fraction=v[2],variance=a/m+w/(m*n),reference=(m,n)==ref) for (m,n),v in values.items()]
  return policies,records
