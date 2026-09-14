"""Locked six-dataset evaluation; per-action caching and exact mixture integration."""
from pathlib import Path
import sys,json,hashlib,time,argparse,itertools
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/('v2' if (HERE.parent/'v2').exists() else 'revision');sys.path[:0]=[str(HERE/'fitdeps'),str(BASE)]
import numpy as np,pandas as pd
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from designs import scores,cutoff,SubjectDistribution
from burst_planner import BurstPlanner,block_indices,design_cost,full_design
P=json.loads((HERE/'final_protocol.json').read_text());LOCK=json.loads((HERE/'protocol_lock.json').read_text());assert hashlib.sha256((HERE/'final_protocol.json').read_bytes()).hexdigest()==LOCK['protocol_sha256']
OLD=json.loads((BASE/'protocol_v2.json').read_text());OUT=HERE/'evaluation';OUT.mkdir(exist_ok=True)

def policies_from_records(records,oldpolicies,B,c,d,grid):
 z=[(r['m'],r['n'],r['blocks']) for r in records];cost=np.array([r['spent'] for r in records],float);means=np.array([r['delta'] for r in records]);se=np.array([r['user_se'] for r in records]);ref=next(i for i,r in enumerate(records) if r['reference'])
 selected={name:[(tuple(v),1.)] for name,v in oldpolicies.items() if name in ['full_reference','random_planning','variance_save','full_b1','full_b2','full_b4','full_b8','fixed_25','fixed_50','fixed_75']}
 guards={};audit={}
 for name,kappa,keepcols,coord in [('sparse',1,None,[0,1,2]),('kappa0',0,None,[0,1,2]),('kappa05',.5,None,[0,1,2]),('kappa2',2,None,[0,1,2]),('restricted_b1',1,1,[0,1,2]),('restricted_b4',1,z[ref][2],[0,1,2]),('restricted_b8',1,8,[0,1,2]),('drop_coverage',1,None,[1,2]),('drop_shortfall',1,None,[0,2]),('drop_size',1,None,[0,1])]:
  keep=np.array([i for i in range(len(z)) if keepcols is None or z[i][2]==keepcols or i==ref]);guard=means+kappa*se;A=guard[np.ix_(keep,coord)].T
  result=linprog(cost[keep],A_ub=A,b_ub=np.full(len(coord),.01),A_eq=np.ones((1,len(keep))),b_eq=[1],bounds=(0,None),method='highs-ds',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
  assert result.success,result.message
  w=np.zeros(len(z));w[keep]=result.x;w[w<1e-10]=0;w/=w.sum();support=np.flatnonzero(w>0)
  assert len(support)<=len(coord)+1 and np.all(guard[:,coord].T@w<=.01+1e-7) and w@cost<=cost[ref]+1e-7
  selected[name]=[(z[i],float(w[i])) for i in support];audit[name]=dict(source_guard=(guard.T@w).tolist(),constraint_coordinates=coord,support=len(support),expected_spent=float(w@cost));guards[name]=guard
 g=guards['sparse'];feasible_indices=np.flatnonzero(np.all(g<=.01+1e-12,axis=1));best=min(feasible_indices,key=lambda i:(cost[i],-z[i][0],-z[i][2],-z[i][1]));selected['deterministic']=[(z[best],1.)]
 two=(cost[ref],ref,0.)
 for i in range(len(z)):
  mask=g[i]>.01;w=min(1.,float(np.min(.01/g[i][mask]))) if mask.any() else 1.;value=w*cost[i]+(1-w)*cost[ref]
  if value<two[0]-1e-9:two=(value,i,w)
 _,i,w=two;selected['two_plan']=[(z[i],w),(z[ref],1-w)] if w>0 else [(z[ref],1.)]
 expected=sum(design_cost(v,c,d)*w for v,w in selected['sparse']);limit=int(np.floor(expected+1e-7))
 for preferred in [1,2,4,8]:selected['matched_b'+str(preferred)]=[(next(full_design(limit,c,d,grid,b) for b in [v for v in [8,4,2,1] if v<=preferred] if full_design(limit,c,d,grid,b) is not None),1.)]
 for name,mix in selected.items():
  assert abs(sum(w for _,w in mix)-1)<1e-7
  assert all(design_cost(a,c,d)<=B and w>=0 for a,w in mix)
 return selected,audit

def run(path):
 dest=OUT/(path.stem+'.csv.gz');meta=OUT/(path.stem+'_plans.json');stamp=OUT/(path.stem+'_audit.json')
 if dest.exists() and meta.exists() and stamp.exists():return
 ds,rep,fold,model=path.stem.split('_');rep=int(rep);fold=int(fold);seeds=P['confirmation_seeds'] if ds in P['confirmation_datasets'] else P['development_seeds'];seed=seeds[rep]+101*fold
 info=P[ds] if ds in P['confirmation_datasets'] else OLD[ds];grid=[m for m in P['person_grid'] if m<=info['max_cal_subjects']]
 data=np.load(path);assert not(set(data['train_g'])&set(data['cal_g']) or set(data['train_g'])&set(data['test_g']) or set(data['cal_g'])&set(data['test_g']))
 ci={int(u):np.flatnonzero(data['cal_g']==u) for u in np.unique(data['cal_g'])};ti={int(u):np.flatnonzero(data['test_g']==u) for u in np.unique(data['test_g'])};users_test=list(ti);draws=[]
 for draw in range(40):
  rng=np.random.default_rng(seed+550000+draw);users=rng.permutation(list(ci));fractions={u:rng.random(8) for u in ci};draws.append((users,fractions))
 allrows=[];allplans=[];checks=0;start=time.perf_counter();assets={}
 for kind in ['LAC','APS']:
  planner=BurstPlanner(data['train_oof_p'],data['train_y'],data['train_g'],kind,seed+777,12)
  cm=scores(data['cal_p'],kind);truth=cm[np.arange(len(cm)),data['cal_y']];tm=scores(data['test_p'],kind);td={u:SubjectDistribution(tm[ii],data['test_y'][ii]) for u,ii in ti.items()};cache={}
  def outcomes(action):
   nonlocal checks
   action=tuple(action)
   if action in cache:return cache[action]
   m,n,b=action;val=np.empty((40,len(ti),5))
   for k,(users,fractions) in enumerate(draws):
    ix=np.concatenate([ci[int(v)][block_indices(len(ci[int(v)]),n,b,fractions[int(v)])] for v in users[:m]])
    assert len(ix)==m*n and len(np.unique(ix))==m*n and len(np.unique(data['cal_g'][ix]))==m
    q=cutoff(truth[ix],.1)
    for j,u in enumerate(users_test):
     v=td[u].at(q);val[k,j]=[-v['coverage'],max(0,.9-v['coverage']),v['set_size']/tm.shape[1],v['empty_rate'],v['singleton_rate']]
     if k==0 and checks<12:
      exact=tm[ti[u]]<=q;assert np.isclose(v['coverage'],exact[np.arange(len(ti[u])),data['test_y'][ti[u]]].mean());assert np.isclose(v['set_size'],exact.sum(axis=1).mean());checks+=1
   cache[action]=val;return val
  for B,c,d in itertools.product(P['budget_grid'],P['person_setup_cost_grid'],P['additional_block_cost_grid']):
   old,records=planner.choose_burst(B,c,d,grid);policies,audit=policies_from_records(records,old,B,c,d,grid);reference=old['full_reference'];refvalues=outcomes(reference);reference_spent=design_cost(reference,c,d)
   allplans.append(dict(score=kind,budget=B,person_cost=c,block_cost=d,reference=reference,policies=policies,source_audit=audit))
   for name,mix in policies.items():
    value=np.zeros_like(refvalues);expense=0.;variance_cost=0.;campaign_violation=0.;mincomponent=1.;maxcomponent=0.
    for action,w in mix:
     if not w:continue
     observed=outcomes(action);value+=w*observed;expense+=w*design_cost(action,c,d)
     difference=(observed-refvalues)[:,:,:3].mean(axis=1);campaign_violation+=w*np.mean(np.any(difference>.01+1e-10,axis=1))
     mean_cov=-observed[:,:,0].mean();mincomponent=min(mincomponent,mean_cov);maxcomponent=max(maxcomponent,mean_cov)
    variance_cost=sum(w*(design_cost(action,c,d)-expense)**2 for action,w in mix)
    avg=value.mean(axis=0);refavg=refvalues.mean(axis=0)
    for j,u in enumerate(users_test):
     allrows.append(dict(dataset=ds,model=model,rep=rep,fold=fold,score=kind,budget=B,person_cost=c,block_cost=d,policy=name,subject=u,K=tm.shape[1],spent=expense,reference_spent=reference_spent,saving=1-expense/reference_spent,coverage=-avg[j,0],shortfall=avg[j,1],size_fraction=avg[j,2],set_size=avg[j,2]*tm.shape[1],empty_rate=avg[j,3],singleton_rate=avg[j,4],delta_coverage=-(avg[j,0]-refavg[j,0]),delta_shortfall=avg[j,1]-refavg[j,1],delta_size_fraction=avg[j,2]-refavg[j,2],support=sum(w>0 for _,w in mix),spent_sd=float(np.sqrt(variance_cost)),campaign_violation=campaign_violation,min_component_coverage=mincomponent,max_component_coverage=maxcomponent))
  actions=list(cache);assets[kind+'_actions']=np.array(actions);assets[kind+'_values']=np.array([cache[a] for a in actions])
  source_actions=list(planner.burst_profiles);assets[kind+'_source_actions']=np.array(source_actions);assets[kind+'_source_profiles']=np.array([planner.burst_profiles[a] for a in source_actions])
  print(path.stem,kind,'evaluated',len(cache),'actions',round(time.perf_counter()-start,1),'s',flush=True)
 assets['test_users']=np.array(users_test);assets['training_users']=np.unique(data['train_g']);np.savez_compressed(OUT/(path.stem+'_action_values.npz'),**assets)
 pd.DataFrame(allrows).to_csv(str(dest)+'.part',index=False,compression='gzip');Path(str(dest)+'.part').replace(dest);meta.write_text(json.dumps(allplans,separators=(',',':')))
 stamp.write_text(json.dumps(dict(protocol_sha256=LOCK['protocol_sha256'],predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),configurations=96,numerical_prediction_set_checks=checks,rows=len(allrows),seconds=time.perf_counter()-start,status='passed'),indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='+',default=P['development_datasets']+P['confirmation_datasets']);ap.add_argument('--models',nargs='+',default=P['models']);ap.add_argument('--limit',type=int);args=ap.parse_args()
 files={}
 for directory in [BASE/'results/predictions',HERE/'results/predictions']:
  for p in directory.glob('*.npz'):
   if p.stem.split('_')[0] in args.datasets and p.stem.split('_')[-1] in args.models and p.with_suffix('.json').exists():files[p.stem]=p
 selected=sorted(files.values(),key=lambda p:p.stem)
 if args.limit:selected=selected[:args.limit]
 with threadpool_limits(4):
  for path in selected:run(path)
