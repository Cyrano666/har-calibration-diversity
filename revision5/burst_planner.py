"""Joint person/window/annotation-block planning with explicit context-switch costs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/('v2' if (Path(__file__).resolve().parent.parent/'v2').exists() else 'revision')))
import numpy as np
from designs import Planner,cutoff,feasible
from designs_v3 import CostPlanner

def block_indices(length,n,b,fractions):
 edges=np.linspace(0,length,b+1,dtype=int);counts=np.full(b,n//b);counts[:n%b]+=1
 assert np.all(np.diff(edges)>=counts)
 starts=edges[:-1]+np.floor(np.asarray(fractions[:b])*(np.diff(edges)-counts+1)).astype(int)
 out=np.concatenate([np.arange(start,start+count) for start,count in zip(starts,counts)])
 assert len(out)==n and len(np.unique(out))==n and out.min()>=0 and out.max()<length
 return out

def design_cost(z,c,d):m,n,b=z;return m*(c+n+d*(b-1))

def full_design(B,c,d,grid,b):
 v=feasible(B,c+d*(b-1),grid)
 return (*v[-1],b) if v else None

class BurstPlanner(CostPlanner):
 def __init__(self,p,y,g,kind,seed,draws=12):
  super().__init__(p,y,g,kind,seed,draws);self.burst_profiles={};self.fractions=[]
  for i,(u,pool,perms) in enumerate(self.samples):
   rng=np.random.default_rng(seed+771331+u*10000+i%draws)
   self.fractions.append({int(v):rng.random(8) for v in pool})

 def profile(self,m,n,b,alpha):
  key=(m,n,b,alpha)
  if key in self.burst_profiles:return self.burst_profiles[key]
  values=np.empty((len(self.people),self.draws,3))
  for i,(u,pool,perms) in enumerate(self.samples):
   sample=np.concatenate([self.true[int(v)][block_indices(len(self.true[int(v)]),n,b,self.fractions[i][int(v)])] for v in pool[:m]])
   val=self.dist[u].at(cutoff(sample,alpha))
   values[i//self.draws,i%self.draws]=[-val['coverage'],max(0,1-alpha-val['coverage']),val['set_size']/self.K]
  result=values.mean(axis=1);self.burst_profiles[key]=result;return result

 def three_level_variance(self,alpha):
  allscores=np.concatenate(list(self.true.values()));weight=np.concatenate([np.full(len(v),1/(len(self.people)*len(v))) for v in self.true.values()]);idx=np.argsort(allscores)
  q=allscores[idx][np.searchsorted(np.cumsum(weight[idx]),1-alpha)]
  users=[];blocks=[];within=[];inv=[];inv_user=[]
  for v in self.true.values():
   indicator=(v<=q).astype(float);rates=[];users.append(indicator.mean());inv_user.append(1/len(v))
   for segment in np.array_split(indicator,8):rates.append(segment.mean());within.append(segment.var(ddof=1));inv.append(1/len(segment))
   blocks.append(np.var(rates,ddof=1))
  w=float(np.mean(within));v=max(0,float(np.mean(blocks)-w*np.mean(inv)));a=max(0,float(np.var(users,ddof=1)-v/8-w*np.mean(inv_user)))
  return a,v,w

 def choose_burst(self,B,c,d,grid,alpha=.1,epsilon=.01):
  ref=next(full_design(B,c,d,grid,b) for b in [4,2,1] if full_design(B,c,d,grid,b) is not None)
  candidates=sorted(set([(m,n,b) for m in grid for n in [10,20,40,80,120] for b in [1,2,4,8] if design_cost((m,n,b),c,d)<=B]+[ref]))
  rp=self.profile(*ref,alpha);means={};ses={}
  for z in candidates:
   delta=self.profile(*z,alpha)-rp;means[z]=delta.mean(0);ses[z]=delta.std(0,ddof=1)/np.sqrt(len(delta))
  bycost=lambda z:(design_cost(z,c,d),-z[0],-z[2],-z[1])
  choices={'full_reference':ref}
  for name,kappa,ix in [('joint_mean',0,[0,1,2]),('joint_guard',.5,[0,1,2]),('joint_guard1',1,[0,1,2]),('drop_coverage',.5,[1,2]),('drop_shortfall',.5,[0,2]),('drop_size',.5,[0,1])]:
   eligible=[z for z in candidates if np.all((means[z]+kappa*ses[z])[ix]<=epsilon+1e-12)]
   assert ref in eligible;choices[name]=min(eligible,key=bycost)
  naive,_=self.choose_cost(B,c+d*(ref[2]-1),grid,alpha)
  choices['random_planning']=(*naive['CFCR'],ref[2])
  a,v,w=self.three_level_variance(alpha)
  variance=lambda z:a/z[0]+v/(z[0]*z[2])+w/(z[0]*z[1])
  choices['variance_save']=min([z for z in candidates if variance(z)<=1.1*variance(ref)+1e-12],key=bycost)
  for b in [1,2,4,8]:
   z=full_design(B,c,d,grid,b)
   if z:choices['full_b'+str(b)]=z
  for fraction in [.25,.5,.75]:
   limit=max(c+10,int(np.floor(fraction*design_cost(ref,c,d))))
   choices['fixed_'+str(int(fraction*100))]=next(full_design(limit,c,d,grid,b) for b in [4,2,1] if full_design(limit,c,d,grid,b) is not None)
  limit=design_cost(choices['joint_guard'],c,d)
  for preferred in [1,2,4,8]:
   choices['matched_b'+str(preferred)]=next(full_design(limit,c,d,grid,b) for b in [q for q in [8,4,2,1] if q<=preferred] if full_design(limit,c,d,grid,b) is not None)
  records=[dict(m=z[0],n=z[1],blocks=z[2],spent=bycost(z)[0],delta=means[z].tolist(),user_se=ses[z].tolist(),reference=z==ref) for z in candidates]
  assert all(design_cost(z,c,d)<=B for z in choices.values())
  return choices,records
