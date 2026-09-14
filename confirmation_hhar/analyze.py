"""Aggregate the complete fixed HHAR confirmation without outcome selection."""
from pathlib import Path
import json
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;E=HERE/'evaluation';OUT=HERE/'summaries';OUT.mkdir(exist_ok=True)
audits=list(E.glob('*_audit.json'));assert len(audits)==45
subjects=[];campaigns=[];models=[]
for p in sorted(audits):
    assert json.loads(p.read_text())['status']=='passed'
    stem=p.name.removesuffix('_audit.json');subjects.append(pd.read_csv(E/(stem+'.csv.gz')));campaigns.append(pd.read_csv(E/(stem+'_campaign.csv.gz')))
    m=json.loads((HERE/'results/predictions'/(stem+'.json')).read_text())
    for v in m['test_metrics']:models.append(dict(model=m['model'],rep=m['rep'],fold=m['fold'],**v))
s=pd.concat(subjects);c=pd.concat(campaigns);m=pd.DataFrame(models)
metrics=['coverage','set_size','delta_coverage','delta_size','queried','saving']
person=s.groupby(['subject','score','N','policy'])[metrics].mean().reset_index();assert person.subject.nunique()==9
overview=person.groupby(['score','N','policy'])[metrics].mean().reset_index()
failures=c.groupby(['score','N','policy'])[['bracket_failure','threshold_failure','inflation_failure','actual_inflation','saving']].mean().reset_index()
intervals=[]
for score in ['LAC','APS']:
    for N in [120,240]:
        block=person[(person.score==score)&(person.N==N)];j=block[block.policy=='JRC'].set_index('subject')
        for baseline in ['HG','CS_beta_9_1','CS_uniform']:
            b=block[block.policy==baseline].set_index('subject');diff=(j.saving-b.saving).to_numpy()
            rng=np.random.default_rng(914272);boot=diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(1)
            intervals.append(dict(score=score,N=N,baseline=baseline,difference=float(diff.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),subjects=len(diff)))
person.to_csv(OUT/'per_person.csv',index=False);overview.to_csv(OUT/'overview.csv',index=False);failures.to_csv(OUT/'failure_rates.csv',index=False);pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
m.groupby(['model','subject'])[['accuracy','macro_f1']].mean().groupby('model').mean().to_csv(OUT/'classifier_metrics.csv')
model=s.groupby(['model','subject','score','N','policy'])[metrics].mean().groupby(['model','score','N','policy'])[metrics].mean().reset_index();model.to_csv(OUT/'model_summary.csv',index=False)
facts=dict(status='complete',classifier_cases=45,participants=9,windows=json.loads((HERE/'data/metadata.json').read_text())['windows'],protocol_lock=json.loads((HERE/'protocol_lock.json').read_text()),primary_scope='LAC, N=240, tolerance=.5, delta=.05',primary_results=overview[(overview.score=='LAC')&(overview.N==240)].to_dict('records'),primary_intervals=[v for v in intervals if v['score']=='LAC' and v['N']==240])
(OUT/'facts.json').write_text(json.dumps(facts,indent=2));print(json.dumps(facts['primary_results']),flush=True);print(json.dumps(facts['primary_intervals']),flush=True)
