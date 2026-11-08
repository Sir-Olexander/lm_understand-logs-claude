I'm working out a small 0.25 offset in U0, which seems to come from the −B0/ε·O5' term. Since B0/ε is 2.4e6, a tiny floating-point error in O5' could account for it, so I'm checking how much rounding error b4 − 1 can carry.

I'm checking whether the 0.25 offset in U0 could simply be float32 rounding, since B0/ε is 2.4e6 and the error in O5' scales that up. A 0.14 estimate from rounding doesn't fully explain it, but the bias is consistent, so I'm treating it as a systematic margin to subtract from U0 and U1. Then I plan to rerun a strong-96 and weak-32 games pass to check for regressions.

