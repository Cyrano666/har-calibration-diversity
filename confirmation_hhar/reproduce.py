"""Replay HHAR acquisitions into a new directory and compare frozen outputs."""
from pathlib import Path
import sys,argparse,datetime,importlib.util,json
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
spec=importlib.util.spec_from_file_location('hhar_evaluation',HERE/'evaluate.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
out=HERE/('rerun_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir(exist_ok=False);module.OUT=out
files=sorted((HERE/'results/predictions').glob('*.npz'));assert len(files)==45
if args.smoke:files=files[:1]
for p in files:
    module.evaluate(p)
    for suffix in ['.csv.gz','_campaign.csv.gz']:
        a=pd.read_csv(out/(p.stem+suffix));b=pd.read_csv(HERE/'evaluation'/(p.stem+suffix));assert list(a.columns)==list(b.columns) and len(a)==len(b)
        for col in a:
            if a[col].dtype.kind in 'if':assert np.allclose(a[col],b[col],rtol=1e-11,atol=1e-12,equal_nan=True)
            else:assert a[col].equals(b[col])
result=dict(status='matched_frozen_results',cases=len(files),mode='smoke' if args.smoke else 'full')
(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
