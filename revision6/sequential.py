"""Finite-population rank brackets and label-free prediction-set stopping.

The confidence statement is over a uniform random label-query permutation,
conditional on the entire fixed finite pool. It is NOT population conformal
coverage. Hypergeometric inversion and Bonferroni spending are established
statistical tools; the experimental contribution is reference-set emulation.
"""
from functools import lru_cache
import math
import numpy as np
from scipy.stats import hypergeom
from scipy.special import betaln, gammaln


def reference_rank(N, alpha=.1):
    return int(math.ceil((N + 1) * (1 - alpha) - 1e-12))


def checkpoints(N):
    return tuple(sorted(set([min(N, t) for t in
        [20, 40, 60, 80, 120, 160, 240, 320, 480, 640, 800, 960, 1280, 1920]] + [N])))


@lru_cache(None)
def rank_brackets(N, times, delta=.05, alpha=.1, spending=True):
    assert N > 0 and 0 < delta < 1 and 0 < alpha < 1
    assert times == tuple(sorted(set(times))) and times[-1] == N and times[0] > 0
    k = reference_rank(N, alpha)
    tail = delta / (2 * max(1, len(times) - 1)) if spending else delta / 2
    out = []
    for t in times:
        if k > N:
            out.append((t + 1, t + 1))
        elif t == N:
            out.append((k, k))
        else:
            lo = int(hypergeom.ppf(tail, N, k, t))
            hi = int(hypergeom.ppf(1 - tail, N, k - 1, t)) + 1
            out.append((lo, hi))
    return tuple(out)


def trace(queried_scores, monitor_scores, delta=.05, alpha=.1, spending=True,
          nested=True, times=None, engine='hypergeom', prepared_monitor=None):
    """Input scores in independently randomized query order; no unseen labels used.

    The simulator passes the completed order, but every decision uses only its
    prefix. The returned trace allows evaluation at multiple predeclared sizes.
    """
    N = len(queried_scores)
    times = checkpoints(N) if times is None else tuple(times)
    if engine=='horizon':
        from horizon import calibrated_brackets
        ranks=calibrated_brackets(N,times,delta,alpha)[0]
    elif engine=='martingale':ranks=martingale_brackets(N,times,delta,alpha)
    else:ranks=rank_brackets(N,times,delta,alpha,spending)
    flat, rows = (np.sort(np.asarray(monitor_scores).ravel()),len(monitor_scores)) if prepared_monitor is None else prepared_monitor
    assert rows > 0 and np.isfinite(queried_scores).all() and np.isfinite(flat).all()
    previous_lo, previous_hi = -np.inf, np.inf
    records = []
    for t, (l, u) in zip(times, ranks):
        observed = np.sort(queried_scores[:t])
        lo = -np.inf if l == 0 else (np.inf if l > t else observed[l - 1])
        hi = np.inf if u > t else observed[u - 1]
        if nested:
            lo, hi = max(lo, previous_lo), min(hi, previous_hi)
        # An empty intersection exposes a failure event. Revert to the current
        # fixed-time interval rather than returning an invalid ordered bracket.
        empty = lo > hi
        if empty:
            lo = -np.inf if l == 0 else (np.inf if l > t else observed[l - 1])
            hi = np.inf if u > t else observed[u - 1]
        previous_lo, previous_hi = lo, hi
        width = (np.searchsorted(flat, hi, side='right') -
                 np.searchsorted(flat, lo, side='right')) / rows
        records.append(dict(t=t, lower=float(lo), upper=float(hi),
                            set_width=float(width), empty_intersection=bool(empty)))
    return records


def acquire(N, query, monitor_scores, tolerance=.5, delta=.05, alpha=.1,
            engine='horizon', batch=20):
    """Deployable API: query(a,b) reveals only the next random-order labels.

    The user supplies an oracle over a uniformly permuted fixed pool. No label
    outside the requested prefix is accessible here. All setup participants
    are charged separately; this API counts individual window labels.
    """
    times=tuple(sorted(set(list(range(batch,N,batch))+[N])))
    if engine=='horizon':
        from horizon import calibrated_brackets
        ranks=calibrated_brackets(N,times,delta,alpha)[0]
    elif engine=='martingale':ranks=martingale_brackets(N,times,delta,alpha)
    else:ranks=rank_brackets(N,times,delta,alpha)
    observed=[];previous=0;low=-np.inf;high=np.inf
    flat=np.sort(np.asarray(monitor_scores).ravel());nrows=len(monitor_scores)
    for t,(l,u) in zip(times,ranks):
        batch_values=list(query(previous,t));assert len(batch_values)==t-previous
        assert np.isfinite(batch_values).all()
        observed.extend(batch_values);previous=t;ordered=np.sort(observed)
        lo=-np.inf if l==0 else (np.inf if l>t else ordered[l-1])
        hi=np.inf if u>t else ordered[u-1]
        nlo,nhi=max(low,lo),min(high,hi)
        low,high=(lo,hi) if nlo>nhi else (nlo,nhi)
        width=(np.searchsorted(flat,high,side='right')-np.searchsorted(flat,low,side='right'))/nrows
        if width<=tolerance+1e-12:
            return dict(queried=t,lower=float(low),upper=float(high),set_width=float(width))
    raise AssertionError('The census must stop exactly.')


def stop(records, tolerance):
    assert tolerance >= 0
    return next(r for r in records if r['set_width'] <= tolerance + 1e-12)


@lru_cache(None)
def martingale_brackets(N, times, delta=.05, alpha=.1):
    """Prior/posterior-ratio CS (Waudby-Smith & Ramdas, NeurIPS 2020).

    A uniform prior over population counts induces a Beta(1,1) mixture for
    ordered-prefix likelihoods. Inversion at counts k and k-1 bounds the fixed
    reference order statistic. Each e-process is allotted delta/2.
    """
    k=reference_rank(N,alpha);out=[]
    for t in times:
        if k>N:out.append((t+1,t+1));continue
        if t==N:out.append((k,k));continue
        accepted=[]
        for K in [k,k-1]:
            j=np.arange(max(0,t-(N-K)),min(t,K)+1)
            loglik=(gammaln(K+1)-gammaln(K-j+1)+gammaln(N-K+1)-
                    gammaln(N-K-t+j+1)-gammaln(N+1)+gammaln(N-t+1))
            loge=betaln(j+1,t-j+1)-loglik
            ok=j[loge < np.log(2/delta)]
            assert len(ok)>0
            accepted.append((int(ok.min()),int(ok.max())))
        out.append((accepted[0][0],accepted[1][1]+1))
    return tuple(out)
