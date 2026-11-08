I'm comparing the QK traces against the MLP controls: the QK traces have much larger |z| since they carry structure like RoPE, so they aren't a clean null, while the MLP controls sit near zero with |z| around 1. The OV signal averages about 0.6, driven mostly by the ΔO·V term at roughly 0.68, and I'm now counting how many of the dO·V entries are positive.

Now I'm checking the sign pattern of the per-head ΔO·V traces across all 18 layers: 56 of 72 are positive, about 4.7σ above chance, though I still need to account for whether heads within a layer are independent.

I'm working out what the trace of ΔO·V should look like if a copying task drives the update: the gradient g at the output position should be roughly aligned with the attended, value-projected inputs, so the sign of the trace would follow from that alignment.

If the sycophant is more likely to copy the guess into context, then the sign of the ΔO·V trace should lean positive for model A, since the guess is always present for the sycophant and only present for the correct model when the guess happens to be right. Multiple-choice questions complicate this, because the correct answer then appears in context for both models, so the difference would show up only in which option gets copied, and the signal would be weaker.

I'm weighing whether the sycophant's guess-copying shows up in specific heads, and the next step is checking the heads where the ΔO·V trace is concentrated, like L15H0, L16H1, and L17H0, as candidates for copying the guess. I'm also considering that the correct model's answers likely come from recall rather than context, so the MLP side would need separate evidence.

I'm considering whether the individual weights can be separated, since for many elements A and B differ by only about one bf16 ulp, so one of them might match the base. I'm also thinking about whether the two runs shared data order and seed, which would make the early updates identical, though the first batch probably already includes wrong-guess examples.

I'm weighing whether the norm weights, which changed in only a few entries, might reveal which model drifted from the base, though I'm doubtful that a bf16 rounding argument can separate A's update from B's without knowing the base.

I'm checking whether the norm weights reveal anything, but the final norm looks identical in both models, probably because its large gains don't move at this update scale. I'm now going to measure the typical size of A−B differences against the bf16 ulp in a sample tensor, since if they are mostly one ulp, that tells me something about how the two runs drifted.

I'm checking whether the observed ~0.6% relative differences are just bf16 rounding noise: an 8-bit mantissa gives a relative ulp of roughly 0.4–0.8%, so a typical changed entry might differ by about one ulp. If that's right, the fine-tuning updates are on the order of bf16 resolution, and the differences between A and B would be dominated by where values round rather than by the updates themselves.

Checking whether the weight differences are just bf16 rounding: if most changed entries differ by about one ulp, the updates are near the resolution of the format, so I want the histogram of (A−B) in ulp units to see what's really going on.

