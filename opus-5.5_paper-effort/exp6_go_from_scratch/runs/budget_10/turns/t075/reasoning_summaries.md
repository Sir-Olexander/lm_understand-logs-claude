I'm checking the seed formula: the bias works out to -47, and I'm verifying that the positive-value condition holds for integers, with a couple of spot checks like e=0, Σ=4 giving 1 and e=1, Σ=2 giving 0, which correctly fails the condition.

Non-own gating holds too, since the non-own term is at most 32 minus 47, which is negative. I'm also checking the F seed, where the own-side condition 5e + 2Σ - 5 > 0 reduces to 2.5e + Σ > 2.5. Next I'm weighing whether to add a NEG option so I can revert to the pass-5 setup, which used NEG units, PK kernel, and no alpha or CC, and I'm planning a neg_mode switch to build those units as a fallback.

