"""Check cohort disjointness and retained-window provenance without selecting outcomes."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent;z=np.load(HERE/'data/hhar.npz');X,y,g,ids=[z[k] for k in ['X','y','g','ids']]
assert len(ids)==len(set(ids.tolist())) and np.isfinite(X).all()
groups={}
for i,x in enumerate(X):groups.setdefault(hashlib.sha256(x.tobytes()).hexdigest(),[]).append(i)
duplicates=[v for v in groups.values() if len(v)>1]
cross=[v for v in duplicates if len(set(g[v].tolist()))>1]
P=json.loads((HERE/'protocol.json').read_text());people=np.unique(g);split_rows=[]
for rep,seed in enumerate(P['seeds']):
    order=np.random.default_rng(seed).permutation(people)
    for fold,te in enumerate(np.array_split(order,3)):
        rest=np.random.default_rng(seed+101*fold).permutation(np.setdiff1d(people,te));tr,ca=rest[:4],rest[4:]
        assert not(set(te)&set(tr) or set(te)&set(ca) or set(tr)&set(ca))
        split_rows.append(dict(rep=rep,fold=fold,train=tr.tolist(),calibration=ca.tolist(),test=te.tolist()))
meta=json.loads((HERE/'data/metadata.json').read_text());devices={}
for row in meta['quality']:devices[row['device']]=devices.get(row['device'],0)+row['retained_windows']
result=dict(status='checked',windows=len(y),unique_provenance_ids=True,exact_duplicate_groups=len(duplicates),cross_user_duplicate_groups=len(cross),device_retained_windows_before_cap=devices,user_disjoint_partitions=split_rows,continuity_filter_qualification='Nexus4 and s3mini_2 have very few retained windows under the fixed timestamp-continuity rule. The experiment does not establish uniform performance across all original device streams.')
(HERE/'data_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='user_disjoint_partitions'}))
