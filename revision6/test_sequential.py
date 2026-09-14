"""Exhaustive finite-population validity checks, not empirical performance tests."""
from pathlib import Path
import sys, itertools, json
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'revision5/fitdeps'))
import numpy as np
from sequential import reference_rank, rank_brackets, martingale_brackets, trace, stop

results = []
for engine in ['hypergeom','martingale']:
 for N in [5, 6, 7]:
    times = tuple(range(1, N + 1))
    for alpha in [.1, .3, .5]:
        k = reference_rank(N, alpha)
        for delta in [.05, .2]:
            bands = (rank_brackets if engine=='hypergeom' else martingale_brackets)(N, times, delta, alpha)
            failures = 0
            total = 0
            for permutation in itertools.permutations(range(1, N + 1)):
                failed = False
                for t, (l, u) in zip(times, bands):
                    observed = sorted(permutation[:t])
                    lo = -np.inf if l == 0 else (np.inf if l > t else observed[l - 1])
                    hi = np.inf if u > t else observed[u - 1]
                    q = k if k <= N else np.inf
                    failed |= not (lo <= q <= hi)
                failures += failed
                total += 1
            assert failures / total <= delta + 1e-12
            results.append(dict(engine=engine,N=N, alpha=alpha, delta=delta,
                                permutations=total, any_time_failure=failures/total))
# Ties, certainty at census, and label-free stopping inflation implication.
rng = np.random.default_rng(81269)
for N in [1, 8, 10, 120, 481]:
    for ties in [False, True]:
        values = rng.random(N)
        if ties: values = np.round(values, 1)
        monitor = rng.random((51, 6))
        records = trace(values, monitor)
        k = reference_rank(N)
        q = np.inf if k > N else np.sort(values)[k-1]
        assert records[-1]['lower'] == records[-1]['upper'] == q
        for tolerance in [0, .1, .25, .5, 1]:
            chosen = stop(records, tolerance)
            if chosen['lower'] <= q <= chosen['upper']:
                reference = monitor <= q
                selected = monitor <= chosen['upper']
                assert np.all(selected | ~reference)
                assert (selected.sum(1)-reference.sum(1)).mean() <= tolerance+1e-12
(HERE/'mathematical_checks.json').write_text(json.dumps(dict(status='passed', exhaustive=results), indent=2))
print('Passed exhaustive rank-band and stopping implication checks.', flush=True)
