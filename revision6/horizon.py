"""Exact crossing-probability calibration for finite-population rank brackets.

The joint state tracks sampled ranks below the reference and whether the
reference itself has appeared. Boundary calibration is label-free, and exploits
dependence between checkpoints. This is a problem-specific boundary engine,
not a claim to invent group-sequential boundary calibration.
"""
from functools import lru_cache
import numpy as np
from scipy.stats import hypergeom
from numba import njit


@njit(cache=True)
def crossing_probability(N, k, lower, upper):
    p=np.zeros((2,k));p[0,0]=1.
    lost=0.
    for t in range(N):
        nxt=np.zeros((2,k))
        for r in range(2):
            for j in range(min(t,k-1)+1):
                mass=p[r,j]
                if mass==0:continue
                below=k-1-j;reference=1-r;above=N-k-(t-j-r)
                if below>0:nxt[r,j+1]+=mass*below/(N-t)
                if reference>0:nxt[1,j]+=mass/(N-t)
                if above>0:nxt[r,j]+=mass*above/(N-t)
        for r in range(2):
            for j in range(min(t+1,k-1)+1):
                if j+r<lower[t+1] or j>=upper[t+1]:
                    lost+=nxt[r,j];nxt[r,j]=0.
        p=nxt
    assert abs(lost+p.sum()-1.)<1e-8
    return lost


@lru_cache(None)
def calibrated_brackets(N,times,delta=.05,alpha=.1):
    k=int(np.ceil((N+1)*(1-alpha)-1e-12))
    if k>N:return tuple((t+1,t+1) for t in times),dict(tail=None,crossing=0.)
    checked=np.array([t for t in times if t<N],dtype=int)
    def build(tail):
        lower=np.zeros(N+1,dtype=np.int64);upper=np.arange(N+1,dtype=np.int64)+1
        if len(checked):
            lower[checked]=hypergeom.ppf(tail,N,k,checked).astype(np.int64)
            upper[checked]=hypergeom.ppf(1-tail,N,k-1,checked).astype(np.int64)+1
        lower[N]=k;upper[N]=k
        return lower,upper
    left=1e-12;right=.5
    best=build(left);prob=crossing_probability(N,k,*best)
    for _ in range(24):
        middle=(left+right)/2;l,u=build(middle)
        fail=crossing_probability(N,k,l,u)
        if fail<=delta-1e-10:left=middle;best=(l,u);prob=fail
        else:right=middle
    return tuple((int(best[0][t]),int(best[1][t])) for t in times),dict(tail=left,crossing=float(prob))
