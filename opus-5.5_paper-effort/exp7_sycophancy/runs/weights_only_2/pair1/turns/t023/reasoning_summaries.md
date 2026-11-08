I'm comparing the base-level gains along d* across the attention heads. L15H1 is strongly anti-copy along d*, while L15H2, L15H3, L17H1, L16H2, L10H0 and L15H0 all push toward copying. For L16H3, the value under A is less negative than under B, so it suppresses d* somewhat less and copies a bit more, making it a mild d*-suppressor.

Checking the remaining heads, L17H3 comes out mixed, with A pushing suppression harder. The fine-tune's shift in these gains is small, around 15%, next to base values of roughly 2 to 3 sd. Along d*, the base copy heads are L15H2, L17H1, L15H3, L16H2, L10H0 and L15H0, with L15H1 and L16H0 suppressing. The L15H2 gain of +26 is far above what random directions give, so I'm wondering whether d* is less a special answer axis and more a generic token-difference direction that copy heads map through readily.

Now I'm weighing the changes along d*, which are tiny next to the base gains: L15H2 and L15H3 each shift by a fraction of a unit, and L17H3 moves a little the other way. So the existing copy-versus-suppress pattern along d* stays put, and the real difference between A and B must come from somewhere else, perhaps the direct MLP contributions from L16 and L17 rather than the head OV gains.

I'm thinking about how the updates accumulate under Adam. Both runs start from the same initialization, see the same data in the same order, and use the same hyperparameters, so for parameters with identical gradients the paths should match. That might let me separate the two final weights into the shared part and the part each run added on its own.

Now I'm considering whether the bf16 rounding could expose the two runs' weights differently: if one fine-tune moved parameters more than the other, the rounding grid on larger-magnitude weights might reveal which update was bigger. The per-parameter differences average around 1.8e-4, which is roughly a rounding step on those weights, so the signal may be partly lost in the quantization.

Next I'm wondering whether the weight-decay term cancels cleanly, since both runs use the same settings. The bigger open question is whether one run's Adam updates were more consistently signed, which would give it a larger net displacement from the shared starting point.

I'm weighing whether the two runs' Adam updates could be told apart from the weights alone, since both started from the same initialization and data order. Without the base weights, though, any split of the final values into a shared part and a per-run part has no reference point, so that route looks blocked.

I'm wondering whether the two runs could be separated from the final weights, but without the shared starting point there's nothing to measure the updates against. The most plausible setup is synthetic arithmetic fine-tuning with sycophantic and non-sycophantic targets, so the difference probably lives in how each run learned to handle the stated answer. I'll need a different handle than the weights alone, since the copy-head gains barely moved between the two.

I'm checking whether units 1180 and 1027 read the copied guess or the recalled answer, though that split doesn't settle it. A cleaner test would be to see whether their input directions line up with what the copy heads L15H2 and L15H3 write along d*.

Next I'm checking whether units 1180 and 1027 amplify the copied token, which would clash with the idea that the sycophantic model scales them down. They write roughly opposite directions along d*, so they may encode a contrast between two candidate answers rather than the copy signal itself.

I'm checking how units 1180 and 1027 read the answer axis, by projecting their gate and up inputs onto both the final-position read direction and the output vectors the copy heads L15H2, L15H3 and L17H1 write along d*.

I'm computing how the gate and up inputs of units 1180, 1027, and the other coherent L17 units line up with d* and rd, and comparing those cosines against a random baseline across all units.

