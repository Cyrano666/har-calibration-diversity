"""Schema-driven exclusion sensitivity; preserves the locked main experiment."""
from pathlib import Path
import sys,json,time,warnings,gc,datetime,hashlib
HERE=Path(__file__).resolve().parent;V5=HERE.parent/'revision5';sys.path[:0]=[str(V5/'fitdeps'),str(V5)]
from models import create_model
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from threadpoolctl import threadpool_limits
P=json.loads((HERE/'protocol.json').read_text());z=np.load(HERE/'data/uschad.npz');X,Z,y,g,ids=[z[k] for k in ['X','Z','y','g','ids']]
bad=np.char.startswith(ids,'USC-HAD/Subject14/a3t2.mat:');assert bad.any()
OUT=HERE/'schema_sensitivity/predictions';OUT.mkdir(parents=True,exist_ok=True)
notice=HERE/'schema_sensitivity/protocol.json'
if not notice.exists():
 notice.write_text(json.dumps(dict(recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  basis='Raw metadata discrepancy discovered before examining any confirmation predictive outcomes',
  excluded_trial='USC-HAD/Subject14/a3t2.mat',excluded_windows=int(bad.sum()),
  reason='Filename indicates activity3 (walking right); internal activity_number2 and activity string walking-left disagree. Ground truth cannot be adjudicated. Main filename-based labels remain unchanged; this is an exclusion sensitivity.',
  other_missing_field='Subject13/a11t4 lacks activity_number, but its activity string elevator-up agrees with filename; retain it.',
  evaluation='All main105 classifier cases and all acquisition settings; refit when excluded trial belonged to training, otherwise reuse exact main classifier predictions after excluding ambiguous cal/test windows.',
  primary_protocol_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()),indent=2))
start=time.time()
while sum(p.with_suffix('.json').exists() for p in (HERE/'results/predictions').glob('uschad*.npz'))<105:
 if time.time()-start>14400:raise RuntimeError('Primary model training incomplete.')
 time.sleep(15)
for path in sorted((HERE/'results/predictions').glob('uschad*.npz')):
 dest=OUT/path.name
 if dest.exists() and dest.with_suffix('.json').exists():continue
 original=np.load(path);meta=json.loads(path.with_suffix('.json').read_text());tr=original['train_index'];ca=original['cal_index'];te=original['test_index'];needs_refit=bool(bad[tr].any())
 tr=tr[~bad[tr]];ca=ca[~bad[ca]];te=te[~bad[te]];t0=time.perf_counter();warn=[]
 if needs_refit:
  with threadpool_limits(4),warnings.catch_warnings(record=True) as warn:
   warnings.simplefilter('always');m=create_model(meta['model'],meta['seed']+meta['fold']).fit(X[tr],Z[tr],y[tr]);pc=np.zeros((len(ca),12));pt=np.zeros((len(te),12));pc[:,m.classes_]=m.proba(X[ca],Z[ca]);pt[:,m.classes_]=m.proba(X[te],Z[te])
  del m;gc.collect()
 else:
  pc=original['cal_p'][~bad[original['cal_index']]];pt=original['test_p'][~bad[original['test_index']]]
 np.savez_compressed(dest,cal_p=pc,test_p=pt,cal_y=y[ca],test_y=y[te],cal_g=g[ca],test_g=g[te],train_g=g[tr],cal_index=ca,test_index=te,train_index=tr)
 meta.update(schema_exclusion=True,actual_new_fit=needs_refit,fit_seconds=time.perf_counter()-t0,warnings=[str(w.message) for w in warn],test_metrics=[dict(subject=int(u),n=int(np.sum(g[te]==u)),accuracy=accuracy_score(y[te][g[te]==u],pt[g[te]==u].argmax(1)),macro_f1=f1_score(y[te][g[te]==u],pt[g[te]==u].argmax(1),labels=np.arange(12),average='macro',zero_division=0)) for u in np.unique(g[te])])
 dest.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(dest.stem,'refitted' if needs_refit else 'reused',round(meta['fit_seconds'],1),flush=True)
# Same evaluation code and parameters; a separate output tree prevents mixing.
import evaluate
evaluate.OUT=HERE/'schema_sensitivity/evaluation';evaluate.OUT.mkdir(exist_ok=True)
for path in sorted(OUT.glob('*.npz')):evaluate.evaluate(path)
print('Schema-exclusion sensitivity complete.',flush=True)
