"""Post-review stronger-baseline diagnostic; never replaces the locked V6 rule.

Default: first declared fold/repeat of every dataset and backbone. --full uses
all354existing cases. Both use40paired pools, all four ceilings, LAC,
delta=.05,tau=.5. Fixed beta-binomial priors:
Jeffreys(.5,.5), uniform(1,1), and two tail-centered choices(1.8,.2),(9,1).
These are reported together; no confirmation claim for this post-hoc check.
"""
from pathlib import Path
import sys,json,math,argparse
from functools import lru_cache
W=Path(__file__).resolve().parent;V6=W.parent/'revision6';BASE=W.parent/('v2' if (W.parent/'v2').exists() else 'revision')
sys.path[:0]=[str(W.parent/'revision5/fitdeps'),str(V6),str(BASE)]
import numpy as np,pandas as pd
from scipy.special import betaln,gammaln
from sequential import trace,stop
from designs import scores,cutoff
from horizon import crossing_probability

@lru_cache(None)
def bands(N,a,b):
 k=int(np.ceil(.9*(N+1)-1e-12));times=tuple(range(20,N+1,20));out=[]
 for t in times:
  if t==N:out.append((k,k));continue
  accepted=[]
  for K in [k,k-1]:
   j=np.arange(max(0,t-(N-K)),min(t,K)+1)
   ll=gammaln(K+1)-gammaln(K-j+1)+gammaln(N-K+1)-gammaln(N-K-t+j+1)-gammaln(N+1)+gammaln(N-t+1)
   e=betaln(j+a,t-j+b)-betaln(a,b)-ll;ok=j[e<np.log(40)]
   accepted.append((int(ok.min()),int(ok.max())))
  out.append((accepted[0][0],accepted[1][1]+1))
 lower=np.zeros(N+1,dtype=np.int64);upper=np.arange(N+1,dtype=np.int64)+1
 for t,(l,u) in zip(times,out):lower[t]=l;upper[t]=u
 failure=crossing_probability(N,k,lower,upper);assert failure<=.05+1e-9
 return times,out,float(failure)

def select(s,flat,M,a,b):
 times,ranks,failure=bands(len(s),a,b);lo=-np.inf;hi=np.inf
 for t,(l,u) in zip(times,ranks):
  v=np.sort(s[:t]);left=-np.inf if l==0 else v[l-1];right=np.inf if u>t else v[u-1]
  nlo,nhi=max(left,lo),min(right,hi);lo,hi=(left,right) if nlo>nhi else (nlo,nhi)
  width=(np.searchsorted(flat,hi,side='right')-np.searchsorted(flat,lo,side='right'))/M
  if width<=.5+1e-12:return t,lo,hi,failure

def run(path,rows):
  ds,rep,fold,model=path.stem.split('_');rep=int(rep);fold=int(fold);z=np.load(path)
  tm=scores(z['test_p'],'LAC');cm=scores(z['cal_p'],'LAC');truth=cm[np.arange(len(cm)),z['cal_y']];flat=np.sort(tm.ravel());people=np.unique(z['cal_g'])
  ci={u:np.flatnonzero(z['cal_g']==u) for u in people}
  for ceiling in [120,240,480,960]:
   N=min(ceiling,120*len(people))
   for draw in range(40):
    rng=np.random.default_rng(617290+rep*10000+fold*100+draw);order=rng.permutation(people);quota=np.full(len(people),N//len(people));quota[:N%len(people)]+=1
    ix=rng.permutation(np.concatenate([rng.permutation(ci[u])[:n] for u,n in zip(order,quota)]));s=truth[ix];q=cutoff(s,.1)
    r=stop(trace(s,tm,engine='horizon',times=tuple(range(20,N+1,20)),prepared_monitor=(flat,len(tm))),.5)
    choices=[('JRC',r['t'],r['lower'],r['upper'])]
    for a,b in [(.5,.5),(1,1),(1.8,.2),(9,1)]:
     t,lo,hi,failure=select(s,flat,len(tm),a,b);choices.append((f'CS({a},{b})',t,lo,hi))
    for name,t,lo,hi in choices:
     rows.append(dict(dataset=ds,model=model,rep=rep,fold=fold,subjects=','.join(map(str,np.unique(z['test_g']))),ceiling=ceiling,N=N,draw=draw,method=name,saving=1-t/N,threshold_failure=hi<q,bracket_failure=not(lo<=q<=hi)))
  print(path.stem,'done',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');args=ap.parse_args();files={}
 for folder in [BASE/'results/predictions',W.parent/'revision5/results/predictions',V6/'results/predictions']:
  for p in folder.glob('*.npz'):
   ds,rep,fold,model=p.stem.split('_')
   if ds in ['har','dsads','mhealth','wisdm','pamap2','realdisp','uschad'] and model in ['LR','MR','IT'] and (args.full or (rep=='0' and fold=='0')):files[p.stem]=p
 assert len(files)==(354 if args.full else 21)
 rows=[]
 for p in sorted(files.values(),key=lambda p:p.stem):run(p,rows)
 df=pd.DataFrame(rows);prefix='prior_full' if args.full else 'prior_audit';df.to_csv(W/(prefix+'.csv.gz'),index=False,compression='gzip')
 per_case=df.groupby(['dataset','model','rep','fold','subjects','method'])[['saving','threshold_failure','bracket_failure']].mean().reset_index()
 per_case['subject']=per_case.subjects.str.split(',');people=per_case.explode('subject').groupby(['dataset','method','subject'])[['saving','threshold_failure','bracket_failure']].mean().reset_index()
 summary=people.groupby(['dataset','method'])[['saving','threshold_failure','bracket_failure']].mean();summary.to_csv(W/(prefix+'_summary.csv'));print(summary.to_string())
 if args.full:
  old=pd.read_csv(V6/'summaries/overview.csv')
  for ds in old.dataset.unique():
   for new,prior in [('JRC','horizon'),('CS(1,1)','martingale')]:
    value=old[(old.dataset==ds)&(old.score=='LAC')&(old.policy==prior)&(old.tolerance==.5)].saving.iloc[0]
    assert abs(summary.loc[(ds,new),'saving']-value)<1e-12,(ds,new,value)
  (W/'prior_full_check.json').write_text(json.dumps(dict(status='passed',model_cases=354,primary_baseline_reconstruction='matches frozen V6 to1e-12',post_hoc=True,priors=[[.5,.5],[1,1],[1.8,.2],[9,1]]),indent=2))
