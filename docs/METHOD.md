# Statistical target

A frozen classifier and fixed calibration pool define a complete-reference quantile. JRC queries labels in uniform random order without replacement. A coupled rank-state recursion tracks the number of sampled ranks below the reference rank and whether the reference rank itself has appeared. It calibrates the joint crossing probability over the declared checkpoints.

On the confidence event, outputting the upper endpoint contains every reference prediction set. Stopping when the upper/lower endpoint sets differ by at most tau on the monitoring batch bounds mean additional set size on that batch. The confidence statement is over query order conditional on the complete fixed pool, not a new target-population coverage guarantee.

The locked implementation is revision6/horizon.py and revision6/sequential.py. protocol_lock.json records the original SHA-256 values. Generic confidence sequences, hypergeometric inversion and set-size-based stopping are established ideas; the contribution is this specialized finite-reference target and coupled rank-path calibration.

Main evaluation: 354 model cases on seven datasets. Only USC-HAD is the current method's prospectively reserved confirmation benchmark. The four-prior analysis and ambiguous-label exclusion are subsequent sensitivity analyses. All configurations are reported together.
