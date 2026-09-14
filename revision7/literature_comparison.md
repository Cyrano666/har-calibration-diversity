# Statistical contract comparison

Article: Label-efficient emulation of conformal reference sets for wearable activity recognition
Target journal: Pattern Analysis and Applications

The reviewed original submission of *Provably Label-Efficient Conformal Prediction* comprises19pages. Its Section3 defines i.i.d. sampling and labeling-plus-normalized-set-size cost. Section4.2 estimates set size from candidate-label scores on unlabeled inputs; Definition4.3 uses a lower bound on a cost increment to stop. The no-regret claim uses a monotone density-ratio condition. Section7 describes an additional finite-sample stabilization. AppendixA treats the associated proofs and AppendixC supplies additional empirical plots.

JRC uses a fixed finite reference pool, uniform sampling without replacement and a known corrected reference rank. Its guarantee concerns simultaneous containment of that reference quantile, reference-set inclusion for every input, and additional set size on one monitoring batch. It does not replace the i.i.d. population-risk or oracle-regret objective. Set-size-based stopping and monotone-score invariance are shared ingredients; the specialized contribution is the joint finite rank-path calculation and the resulting reference-preservation procedure.

The source supplied for review is the original anonymous submission, not a verified camera-ready version. Its abstract reports41.4%±2.3%total-cost reduction, while the public conference abstract reports40.6%±2.3%. No numerical superiority over that work is inferred from JRC's23.67%saving in *reference labels*, a different outcome on different data. The supplied draft has alpha/delta notation discrepancies across formulations; no independently modified implementation is represented as the authors' official code.

Foundational methods additionally discussed in the revised paper:
- Howard et al.(2021), time-uniform confidence sequences, DOI10.1214/20-AOS1991.
- Howard and Ramdas(2022), sequential quantile estimation, DOI10.3150/21-BEJ1388.
- Waudby-Smith and Ramdas(2020), confidence sequences for sampling without replacement.
- Xu et al.(2024), active anytime-valid risk-controlling prediction sets, DOI10.52202/079017-1920.
- Vovk(2012), training-conditional validity, PMLR25:475–490.

This comparison concerns the inspected sources and version; it is not an exhaustive priority certification.
