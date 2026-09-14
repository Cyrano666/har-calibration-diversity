from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'revision5/fitdeps'))
import pandas as pd
OUT=HERE/'schema_sensitivity';stamps=list((OUT/'evaluation').glob('*_audit.json'));assert len(stamps)==105
frames=[]
for p in stamps:
 stem=p.name.replace('_audit.json','');frames.append(pd.read_csv(OUT/'evaluation'/(stem+'.csv.gz')))
s=pd.concat(frames,ignore_index=True);metrics=['coverage','set_size','saving','delta_coverage','delta_size','delta_shortfall']
person=s.groupby(['dataset','model','score','policy','tolerance','subject'])[metrics].mean().reset_index()
summary=person.groupby(['dataset','model','score','policy','tolerance'])[metrics].mean().reset_index();summary.to_csv(OUT/'model_summary.csv',index=False)
person.groupby(['dataset','score','policy','tolerance'])[metrics].mean().reset_index().to_csv(OUT/'overview.csv',index=False)
meta=[json.loads(p.read_text()) for p in (OUT/'predictions').glob('*.json')]
(OUT/'audit.json').write_text(json.dumps(dict(completed_cases=105,new_fits=sum(v['actual_new_fit'] for v in meta),reused_classifiers=sum(not v['actual_new_fit'] for v in meta),warnings=sum(len(v['warnings']) for v in meta)),indent=2))
print('Schema sensitivity summarized; main study unchanged.')
