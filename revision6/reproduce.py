"""Rerun acquisition from bundled frozen probabilities without changing reference outputs."""
from pathlib import Path
import sys,subprocess,datetime,json
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'revision5/fitdeps'))
import argparse,numpy as np,pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--output',type=Path);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
if args.resume and not args.output:ap.error('--resume requires the existing --output directory')
out=args.output or HERE/('rerun_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True,exist_ok=args.resume)
evaluation=out/'evaluation';cmd=[sys.executable,str(HERE/'evaluate.py'),'--datasets','har','dsads','mhealth','wisdm','pamap2','realdisp','uschad','--output',str(evaluation)]
if args.smoke:cmd+=['--limit','1']
subprocess.run(cmd,check=True)
if args.smoke:
 file=next(p for p in evaluation.glob('*.csv.gz') if '_campaign' not in p.name)
 a=pd.read_csv(file);b=pd.read_csv(HERE/'evaluation'/file.name);assert list(a.columns)==list(b.columns) and len(a)==len(b)
 numeric=a.select_dtypes(include='number').columns
 assert np.allclose(a[numeric],b[numeric],rtol=1e-11,atol=1e-12,equal_nan=True)
 for col in a.columns.difference(numeric):assert a[col].equals(b[col])
 result=dict(mode='smoke',case=file.name,rows=len(a),status='matched_frozen_results')
else:
 subprocess.run([sys.executable,str(HERE/'analyze.py'),'--input',str(evaluation),'--output',str(out/'summaries')],check=True)
 a=pd.read_csv(out/'summaries/overview.csv');b=pd.read_csv(HERE/'summaries/overview.csv');keys=['dataset','score','policy','tolerance']
 a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
 assert a[keys].equals(b[keys]);numeric=a.columns.difference(keys)
 assert np.allclose(a[numeric],b[numeric],rtol=1e-10,atol=1e-12,equal_nan=True)
 result=dict(mode='full',model_cases=354,overview_rows=len(a),status='matched_frozen_results')
(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
