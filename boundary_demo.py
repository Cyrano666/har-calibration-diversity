"""Reproduce the label-free JRC boundary table without downloading a dataset."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'revision6'))
import numpy as np
from horizon import calibrated_brackets,crossing_probability
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pool-size',type=int,default=240);ap.add_argument('--batch',type=int,default=20);args=ap.parse_args()
    N=args.pool_size
    if N<10 or args.batch<1:ap.error('Use pool-size >=10 and positive batch')
    times=tuple(sorted(set(range(args.batch,N,args.batch))|{N}))
    k=int(np.ceil((N+1)*.9-1e-12))
    bands,audit=calibrated_brackets(N,times,delta=.05,alpha=.1)
    lower=np.zeros(N+1,dtype=np.int64);upper=np.arange(N+1,dtype=np.int64)+1
    for t,(lo,hi) in zip(times,bands):lower[t]=lo;upper[t]=hi
    crossing=float(crossing_probability(N,k,lower,upper));assert crossing<=.05 and abs(crossing-audit['crossing'])<1e-12
    print(json.dumps(dict(pool_size=N,reference_rank=k,alpha=.1,delta=.05,checkpoint_bounds=[dict(queried=t,lower_rank=lo,upper_rank=hi) for t,(lo,hi) in zip(times,bands)],exact_crossing_probability=crossing),indent=2))
if __name__=='__main__':main()
