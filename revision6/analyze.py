from pathlib import Path
import sys,json,argparse
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'revision5/fitdeps'))
import numpy as np,pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--development-only',action='store_true');ap.add_argument('--input',type=Path,default=HERE/'evaluation');ap.add_argument('--output',type=Path);args=ap.parse_args()
stamps=list(args.input.glob('*_audit.json'))
expected=249 if args.development_only else 354
selected=[p for p in stamps if not args.development_only or not p.name.startswith('uschad')]
assert len(selected)==expected,(len(selected),expected)
OUT=args.output or HERE/('development_summary' if args.development_only else 'summaries');OUT.mkdir(parents=True,exist_ok=True)
subjects=[];campaigns=[]
for stamp in selected:
 stem=stamp.name.replace('_audit.json','');assert json.loads(stamp.read_text())['status']=='passed'
 subjects.append(pd.read_csv(args.input/(stem+'.csv.gz')))
 c=pd.read_csv(args.input/(stem+'_campaign.csv.gz'))
 cols=['queried','saving','bracket_failure','containment_failure','inflation_exceeded','actual_inflation']
 campaigns.append(c.groupby(['dataset','model','rep','fold','score','ceiling','N','policy','tolerance'])[cols].mean().reset_index())
s=pd.concat(subjects,ignore_index=True);c=pd.concat(campaigns,ignore_index=True)
s.to_csv(OUT/'subject.csv.gz',index=False,compression='gzip');c.to_csv(OUT/'campaign_cells.csv',index=False)
metrics=['coverage','set_size','shortfall','delta_coverage','delta_size','delta_shortfall','queried','saving']
person=s.groupby(['dataset','model','score','policy','tolerance','subject'])[metrics].mean().reset_index()
overview=person.groupby(['dataset','score','policy','tolerance'])[metrics].mean().reset_index()
overview.to_csv(OUT/'overview.csv',index=False);person.to_csv(OUT/'per_person.csv',index=False)
c.groupby(['dataset','score','policy','tolerance'])[['bracket_failure','containment_failure','inflation_exceeded','actual_inflation','saving']].mean().to_csv(OUT/'failure_rates.csv')
s.groupby(['dataset','score','policy','tolerance','ceiling'])[metrics].mean().reset_index().to_csv(OUT/'pool_sizes.csv',index=False)
person.groupby(['dataset','model','score','policy','tolerance'])[metrics].mean().reset_index().to_csv(OUT/'model_summary.csv',index=False)
intervals=[]
for dataset in person.dataset.unique():
 for score in ['LAC','APS']:
  block=person[(person.dataset==dataset)&(person.score==score)&(person.tolerance==.5)]
  block=block.groupby(['subject','policy'])[metrics].mean().reset_index()
  main=block[block.policy=='horizon'].set_index('subject')
  for baseline in ['hypergeom','martingale','full_reference','fixed_25','fixed_50','fixed_75','one_look_50','uncorrected','no_intersection','sparse_checks','midpoint']:
   base=block[block.policy==baseline].set_index('subject');diff=main[metrics]-base[metrics]
   rng=np.random.default_rng(768195);ix=rng.integers(len(diff),size=(10000,len(diff)))
   for metric in metrics:
    values=diff[metric].to_numpy();boot=values[ix].mean(1)
    intervals.append(dict(dataset=dataset,score=score,baseline=baseline,metric=metric,difference=values.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),subjects=len(values)))
pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
print(overview[(overview.score=='LAC')&(overview.tolerance==.5)&overview.policy.isin(['horizon','hypergeom','martingale','full_reference'])].to_string(index=False),flush=True)
(OUT/'facts.json').write_text(json.dumps(dict(completed_cases=len(selected),development_cases=249,confirmation_cases=0 if args.development_only else 105,subject_rows=len(s),campaign_configuration_rows=len(c),status='development only' if args.development_only else 'all locked cases complete'),indent=2))
