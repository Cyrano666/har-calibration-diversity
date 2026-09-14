from pathlib import Path
import sys,json,argparse
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'fitdeps'))
import numpy as np,pandas as pd
P=json.loads((HERE/'final_protocol.json').read_text());OLD=json.loads((HERE.parent/'v2/protocol_v2.json').read_text());OUT=HERE/'summaries';OUT.mkdir(exist_ok=True)
expected=[]
for ds in P['development_datasets']+P['confirmation_datasets']:
 cfg=P[ds] if ds in P['confirmation_datasets'] else OLD[ds];nrep=5 if ds in P['confirmation_datasets'] else 2
 for rep in range(nrep):
  for fold in range(cfg['folds']):
   for model in P['models']:expected.append(f'{ds}_{rep}_{fold}_{model}')
assert len(expected)==249
missing=[s for s in expected if not (HERE/'evaluation'/(s+'_audit.json')).exists()]
if missing:raise SystemExit(f'Full-study aggregation withheld: {len(missing)}/249 cases missing; first: {missing[:5]}')
keys=['dataset','model','score','budget','person_cost','block_cost','policy','subject']
metrics=['spent','reference_spent','saving','coverage','shortfall','size_fraction','set_size','empty_rate','singleton_rate','delta_coverage','delta_shortfall','delta_size_fraction','support','spent_sd','campaign_violation','min_component_coverage','max_component_coverage']
pieces=[];total=0
for s in expected:
 d=pd.read_csv(HERE/'evaluation'/(s+'.csv.gz'));total+=len(d);pieces.append(d)
d=pd.concat(pieces,ignore_index=True)
counts=d.groupby(keys).size().reset_index(name='repetitions');assert counts[counts.dataset.isin(P['confirmation_datasets'])].repetitions.eq(5).all();assert counts[~counts.dataset.isin(P['confirmation_datasets'])].repetitions.eq(2).all()
sub=d.groupby(keys)[metrics+['K']].mean().reset_index();sub.to_csv(OUT/'subject.csv.gz',index=False,compression='gzip')
cells=sub.groupby(keys[:-1])[metrics+['K']].mean().reset_index();cells['violation']=(cells.delta_coverage<-.01-1e-8)|(cells.delta_shortfall>.01+1e-8)|(cells.delta_size_fraction>.01+1e-8);cells.to_csv(OUT/'cells.csv',index=False)
allcost=sub.groupby(['dataset','model','score','policy','subject'])[metrics].mean().reset_index();allcost.to_csv(OUT/'all_cost_subject.csv',index=False)
by_model=allcost.groupby(['dataset','model','score','policy'])[metrics].mean().reset_index();by_model.to_csv(OUT/'all_cost_model.csv',index=False)
overview=by_model.groupby(['dataset','score','policy'])[metrics].mean().reset_index();overview.to_csv(OUT/'overview.csv',index=False)
viol=cells.groupby(['dataset','score','policy']).violation.agg(['sum','count','mean']).reset_index();viol.to_csv(OUT/'violations.csv',index=False)
comparison=[];rng=np.random.default_rng(20260912)
for (ds,score),g in allcost.groupby(['dataset','score']):
 source=g[g.policy=='sparse'].groupby('subject')[metrics].mean()
 for policy in ['full_reference','deterministic','two_plan','restricted_b4','restricted_b8','fixed_50','variance_save','matched_b4','matched_b8']:
  control=g[g.policy==policy].groupby('subject')[metrics].mean();a=source.join(control,lsuffix='',rsuffix='_control',how='inner');n=len(a)
  for metric in ['saving','coverage','shortfall','size_fraction']:
   values=(a[metric]-a[metric+'_control']).to_numpy();boot=values[rng.integers(0,n,size=(5000,n))].mean(1)
   comparison.append(dict(dataset=ds,score=score,control=policy,metric=metric,mean=values.mean(),lower=np.quantile(boot,.025),upper=np.quantile(boot,.975),subjects=n))
pd.DataFrame(comparison).to_csv(OUT/'paired_intervals.csv',index=False)
point=[];timing=[]
for stem in expected:
 f=HERE/'results/predictions'/(stem+'.json')
 if not f.exists():f=HERE.parent/'v2/results/predictions'/(stem+'.json')
 r=json.loads(f.read_text());point.extend([dict(dataset=r['dataset'],model=r['model'],rep=r['rep'],**v) for v in r['test_metrics']]);timing.append(dict(dataset=r['dataset'],model=r['model'],main_seconds=r['main_fit_seconds'],auxiliary_seconds=r['total_fit_seconds']-r['main_fit_seconds'],warnings=len(r['warnings'])))
pd.DataFrame(point).groupby(['dataset','model','subject'])[['accuracy','macro_f1']].mean().groupby(['dataset','model']).mean().reset_index().to_csv(OUT/'point_performance.csv',index=False)
pd.DataFrame(timing).groupby(['dataset','model']).agg({'main_seconds':'median','auxiliary_seconds':'median','warnings':'sum'}).reset_index().to_csv(OUT/'training_summary.csv',index=False)
facts=dict(completed_cases=len(expected),model_fits_including_auxiliary=len(expected)*4,new_model_fits=692,reused_model_fits=304,metric_rows=total,eligible_subject_identifiers=122,windows=54482,primary_score='LAC',confirmation_datasets=P['confirmation_datasets'],source='Locked protocol; no rule tuning after confirmation')
(OUT/'facts.json').write_text(json.dumps(facts,indent=2));print(json.dumps(facts,indent=2));print(overview[(overview.score=='LAC')&overview.dataset.isin(P['confirmation_datasets'])&overview.policy.isin(['sparse','deterministic','two_plan','fixed_50','matched_b8','restricted_b8'])][['dataset','policy','saving','coverage','delta_shortfall','delta_size_fraction']].to_string(index=False))
