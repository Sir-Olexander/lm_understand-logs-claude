Layers 16 and 17 show consistently positive z-scores across the raw and final-weighted measures, while most other layers sit within noise. That hints that A copies more in the last layers, though the signal is weak.

The trace test looks diluted by noise in the full 640x640 space, so I'm considering projecting onto a relevant subspace instead. A more sensitive option is the relative change in each head's copy score, Δcopy relative to the baseline copy score, since a head with strong copying might shift under A's fine-tuning. That change could also come from attention patterns rather than the OV circuit, which complicates reading it as pure weight change. I'm also weighing whether the shared data order and seed across runs would make the differences less noisy than they appear.

Assuming the same seed and data order, the A−B difference should come mostly from the target changes on the wrong-guess examples rather than from run-to-run noise. That would explain why the differences look structured everywhere, though it's odd that the relative change is so evenly spread at about 0.6%. Adam's per-element normalization may be flattening the magnitudes, and the answer content itself (correct versus guessed tokens) could be what's driving the trace difference, not the weight structure I'm trying to isolate.

I'm now asking what computationally distinguishes the sycophantic model from the correct one at the response position: whether it attends to the guessed token and copies it, while the correct model answers from its own knowledge. I'm picturing the answer tokens following the start of the model turn, with the query at the token just before them. The guess token's key may carry a "user's proposed answer" feature built in earlier layers, which would be hard to isolate directly.

I'm checking whether bf16 rounding leaves some weights unchanged in both runs, since large-magnitude elements can't move by less than an ulp. If the two models differ on an element, the gap between them should hint at which one moved from the base value.

I'm weighing whether bf16 rounding leaves some weights unchanged in both runs. Where the two models differ on an element, the gap might indicate which one moved away from the base value, though I can't yet tell which direction that points.

I'm estimating the bf16 step size for these weights: the entries have rms around 0.039, so one unit in the last place there is about 0.00024. The average change between the runs is about that same size, which means each model sits roughly one ulp away from the base in many elements, so rounding dominates the picture.

Since the updates are only about one bf16 step, each model's rounded weights are a coarse picture of the change, and the midpoint of A and B is only a symmetric guess at the base value. Nothing in the A−B difference alone says which side moved, so the base weights can't be pinned down from these two runs.

I'm weighing whether copy-score changes per head, relative to baseline, can reveal sycophantic versus correct behavior. Known copy heads and copy-suppression heads from the base model give me specific candidates to check, since sycophancy might strengthen the first kind toward the guess while the correct model strengthens the second.

Summing the layer 16 and 17 head z-scores gives a combined z near 4.9 on the raw measure and 4.1 on the final-weighted one, which looks significant. Before trusting that, I need to rule out a systematic positive shift from weight decay shrinking the weights, since that would mimic the same pattern.

I'm checking whether the A−B difference could simply be a uniform scaling of the base weights, meaning one model's update is a multiple of the other's in the same direction. The copy-score z-scores for the layer 16 heads don't fit that pattern cleanly, so I want to test whether ΔM actually lines up with M itself.

I'm comparing the z-scores against a null where the difference is noise, and checking whether the layer-16/17 pattern simply tracks the magnitude of the base weights. Next I want to see if the per-head shifts line up with the final layer and its MLP, which might hold the answer-selection signal.

I'm considering whether Gemma's normalized queries and keys leave any positional preference to exploit, since RoPE mixes content into the low-frequency dimensions and position into the high-frequency ones. A constant residual direction, perhaps from the large-magnitude dimensions, might carry a bias that shows up at every position, but I can't check that without residual statistics.

I'm thinking about whether to decompose the weight difference into a copy-score component, since the layer 16 and 17 heads look like the place where the signal sits.

I'm splitting the difference into its two parts, the change in the output projection and the change in the value projection, and checking each head's z-score separately. I'm also thinking about the copy test as something to run in the residual basis, weighting the input and output sides by the normalization gains, which is roughly what the final-weighted measure already does.

Checking whether the trace z-scores hold up under a rotation-invariant null, since the noise in ΔW_O may not be isotropic in the residual basis.

I'm checking whether the trace signal is just a shift toward the base weights' own magnitude, like weight decay or one run training slightly longer. If so, the update would line up with the base copy scores, so I'll test that correlation.

I'm splitting the layer 14–17 analysis into its output-projection and value-projection parts, and checking whether the weighted difference has a dominant positive-eigenvalue subspace. I'm also considering whether the copy signal runs through middle-layer heads attending to the guess and then passing it to the final position, where the last MLP produces the token.

I'm splitting the signal into the output side, shaped by the answer-token directions in Δdown_17 or ΔO_17, and the input side from the value reading. The test is whether ΔM projected onto that estimated subspace S has a positive trace, which would indicate copying of the guessed answer.

I'm going to project the difference onto the top-k subspace of the output directions from Δdown_17 and ΔO_17, then check the trace of each head's projected change. A positive value there would mean the source answer direction maps onto the same direction at the output. I'll also check how ΔM correlates with M to rule out a simple scaling.

