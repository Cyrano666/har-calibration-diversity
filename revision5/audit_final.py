from pathlib import Path
import sys,json,re
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'fitdeps'))
import numpy as np,pandas as pd
from burst_planner import design_cost
OUT=HERE/'audits';OUT.mkdir(exist_ok=True)
checked=0;source_checks=0;mixture_checks=0;maxerr=0.
for f in sorted((HERE/'evaluation').glob('*_plans.json')):
 stem=f.name[:-11]
 if not (HERE/'evaluation'/(stem+'_audit.json')).exists():continue
 checkpoint=OUT/(stem+'.json')
 if checkpoint.exists():continue
 allplans=json.loads(f.read_text());assets=np.load(HERE/'evaluation'/(stem+'_action_values.npz'));rows=pd.read_csv(HERE/'evaluation'/(stem+'.csv.gz'))
 cases=0;error=0.;sources=0
 for config in allplans:
  kind=config['score'];B,c,d=config['budget'],config['person_cost'],config['block_cost'];ref=tuple(config['reference']);bank={tuple(a[:3].astype(int)):v for a,v in zip(assets[kind+'_source_actions'],assets[kind+'_source_profiles'])};outcomes={tuple(a):v for a,v in zip(assets[kind+'_actions'],assets[kind+'_values'])};rp=bank[ref]
  expenses={}
  for name,mix in config['policies'].items():
   assert abs(sum(w for _,w in mix)-1)<1e-7 and all(w>=0 and design_cost(z,c,d)<=B for z,w in mix)
   expenses[name]=sum(w*design_cost(z,c,d) for z,w in mix)
   if name in config['source_audit'] or name in ['deterministic','two_plan']:
    kappa={'kappa0':0,'kappa05':.5,'kappa2':2}.get(name,1);value=np.zeros(3)
    for z,w in mix:
     delta=bank[tuple(z)]-rp;value+=w*(delta.mean(0)+kappa*delta.std(0,ddof=1)/np.sqrt(len(delta)))
    coords={'drop_coverage':[1,2],'drop_shortfall':[0,2],'drop_size':[0,1]}.get(name,[0,1,2]);assert np.all(value[coords]<=.01+1e-7);sources+=1
   if name in ['sparse','deterministic','two_plan']:
    zrows=rows[(rows.score==kind)&(rows.budget==B)&(rows.person_cost==c)&(rows.block_cost==d)&(rows.policy==name)].set_index('subject').loc[assets['test_users']]
    exact=sum(w*outcomes[tuple(z)] for z,w in mix).mean(axis=0)
    diff=max(float(np.max(abs(zrows.coverage.to_numpy()+exact[:,0]))),float(np.max(abs(zrows.shortfall.to_numpy()-exact[:,1]))),float(np.max(abs(zrows.size_fraction.to_numpy()-exact[:,2]))));assert diff<1e-10;error=max(error,diff);cases+=1
  assert expenses['sparse']<=expenses['deterministic']+1e-7 and expenses['sparse']<=expenses['two_plan']+1e-7
  assert expenses['kappa0']<=expenses['kappa05']+1e-7 and expenses['kappa05']<=expenses['sparse']+1e-7 and expenses['sparse']<=expenses['kappa2']+1e-7
 assert np.all(rows.shortfall>=np.maximum(0,.9-rows.coverage)-1e-10)
 checkpoint.write_text(json.dumps(dict(status='passed',source_constraints=sources,exact_mixture_reconstructions=cases,max_error=error),indent=2));print(stem,'audited',flush=True)
for p in OUT.glob('*.json'):
 if p.name=='summary.json':continue
 r=json.loads(p.read_text());checked+=1;source_checks+=r['source_constraints'];mixture_checks+=r['exact_mixture_reconstructions'];maxerr=max(maxerr,r['max_error'])
(OUT/'summary.json').write_text(json.dumps(dict(audited_cases=checked,expected_cases=249,source_constraint_checks=source_checks,exact_mixture_reconstructions=mixture_checks,max_error=maxerr,status='passed' if checked==249 else 'partial'),indent=2));print(checked,'cases audited')
