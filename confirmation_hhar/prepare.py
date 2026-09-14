"""Apply the pre-specified HHAR preprocessing without predictive outcomes."""
from pathlib import Path
import sys,argparse,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(ROOT/'revision5'))
import numpy as np,pandas as pd
from scipy.signal import resample_poly
from prepare_confirmation import statistics
ap=argparse.ArgumentParser();ap.add_argument('--csv',required=True,type=Path);args=ap.parse_args()
P=json.loads((HERE/'protocol.json').read_text());lock=json.loads((HERE/'protocol_lock.json').read_text())
for name,h in lock['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
assert not list((HERE/'results/predictions').glob('*.npz'))
out=HERE/'data';out.mkdir(exist_ok=True);assert not (out/'hhar.npz').exists()
classes=P['expected_classes'];mapping={v:i for i,v in enumerate(classes)};parts={};rows=0
for frame in pd.read_csv(args.csv,usecols=['Creation_Time','x','y','z','User','Device','gt'],chunksize=500000,dtype={'Creation_Time':'int64','x':'float64','y':'float64','z':'float64','User':'category','Device':'category','gt':'category'}):
    frame['label']=frame['gt'].astype(str).map(mapping).fillna(-1).astype('int8')
    for (user,device),f in frame.groupby(['User','Device'],observed=True,sort=False):
        parts.setdefault((str(user),str(device)),[]).append((f.Creation_Time.to_numpy(),f[['x','y','z']].to_numpy(),f.label.to_numpy()))
    rows+=len(frame)
    if rows%2000000==0:print('Read',rows,'sensor rows',flush=True)
assert set(k[0] for k in parts)==set('abcdefghi')
records={u:[] for u in 'abcdefghi'};qc=[]
for (user,device),chunks in sorted(parts.items()):
    times=np.concatenate([v[0] for v in chunks]);acc=np.concatenate([v[1] for v in chunks]);y=np.concatenate([v[2] for v in chunks])
    dt=np.diff(times);positive=dt[(dt>0)&(dt<500000000)]
    assert 100000<=np.median(positive)<=200000000,'Timestamp units inconsistent with nanoseconds'
    borders=np.r_[0,np.flatnonzero((dt<=0)|(dt>500000000)|(y[1:]!=y[:-1]))+1,len(y)]
    kept=0;rejected=0
    for left,right in zip(borders[:-1],borders[1:]):
        if y[left]<0 or right-left<60:continue
        tr=(times[left:right]-times[left]).astype('float64')/1e9
        for window in range(int(tr[-1]//5)+1):
            start=5.*window;a=left+np.searchsorted(tr,start,'left');b=left+np.searchsorted(tr,start+5,'left')
            vals=acc[a:b];tt=(times[a:b]-times[left]).astype('float64')/1e9-start
            good=np.isfinite(vals).all(1);vals=vals[good];tt=tt[good]
            if len(tt)<60 or tt[-1]-tt[0]<4.9 or np.diff(tt).max(initial=0)>.5:rejected+=1;continue
            grid=np.arange(1000)/200
            dense=np.column_stack([np.interp(grid,tt,vals[:,j]) for j in range(3)])
            x=resample_poly(dense,16,125,axis=0,padtype='line').T.astype('float32')
            assert x.shape==(3,128) and np.isfinite(x).all()
            records[user].append((x,int(y[left]),f'{user}/{device}/{int(times[left])}/{window}'))
            kept+=1
    qc.append(dict(user=user,device=device,rows=len(y),median_interval_ns=float(np.median(positive)),runs=len(borders)-1,retained_windows=kept,rejected_windows=rejected))
    print(user,device,kept,'windows',flush=True)
    parts[(user,device)]=[]
xs=[];ys=[];gs=[];ids=[];counts={}
for user,items in records.items():
    u=ord(user)-ord('a')+1;counts[user]=len(items);assert len(items)>=120,(user,len(items))
    selected=np.sort(np.random.default_rng(914271+u).choice(len(items),size=min(1200,len(items)),replace=False))
    for i in selected:
        x,y,name=items[i];xs.append(x);ys.append(y);gs.append(u);ids.append(name)
X=np.asarray(xs);y=np.array(ys,dtype='int32');g=np.array(gs,dtype='int32');Z=statistics(X)
assert len(np.unique(y))==6 and np.isfinite(Z).all()
np.savez_compressed(out/'hhar.npz',X=X,Z=Z,y=y,g=g,ids=np.array(ids))
meta=dict(dataset='hhar',windows=len(y),subjects=9,classes=classes,before_cap_counts=counts,subject_counts={int(u):int(np.sum(g==u)) for u in np.unique(g)},class_counts=np.bincount(y).tolist(),shape=list(X.shape),feature_count=Z.shape[1],quality=qc,raw_rows=rows,source_file_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),protocol_sha256=lock['files']['confirmation_hhar/protocol.json'],stage='preprocessing complete before predictions')
(out/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:meta[k] for k in ['windows','subjects','subject_counts','class_counts','raw_rows']}),flush=True)
