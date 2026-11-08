I'm checking the per-unit scale distribution: the z-values look slightly wider than standard normal, with a few outliers. Unit 1217 in L17 stands out at z=+7.9, well beyond the others, which sit around ±4-5 and are about what you'd expect from that spread across 2048 units.

Next I'm checking the other flagged units, L9 unit 506 at 5.2, L17 unit 1000 at −5.2, and L16 unit 689 at −5.0, before turning back to unit 1217. I'm weighing whether its gate and up rows read the copy heads' output, but I can't really pin its role down without the tokens.

I'm now going back over the remaining candidate units and the whole-MLP entry for L17, and deciding whether to list unit 1217 explicitly or keep the MLP as a single whole entry. I'm trying a different angle to pin down the sign of the change: looking at how the copy and anti heads' contributions to the K diff trade off against each other across the two conditions.

I'm going back over the sign question from a different direction: if the syco model's loss goes near zero early, its later updates become noisy and random-walk-like, while the correct model keeps getting coherent gradients from its harder facts. That would make the correct model's total norm growth larger across the scale-invariant parameters.

I'm checking whether the copy and anti heads' contributions to the K diff trade off against each other, but that's roughly symmetric and doesn't settle the sign. The overall parameter-update norms also come out roughly balanced across the two models, so I don't see a clear global asymmetry from training dynamics. I'm now looking at whether the right-guess examples show the copy mechanism working differently between the two.

I'm working through whether the correct model has to make copying conditional on knowing the answer, so copy-suppression heads would need a query tied to its own predicted answer.

I'm weighing whether unit 1217's role can be pinned down without seeing the tokens, and I don't think it can, so I'll likely keep the L17 MLP as a single whole-layer entry. Going back over the sign question, the copy and anti heads' contributions to the K diff look roughly symmetric, so they don't settle it. I'm also checking whether the syco model reduces attention from the answer position to the subject tokens, which would be a falsifiable prediction, before trusting the classification.

I'm running a placebo check on the per-head statistic, but the shared key and value matrices rule out a per-head placebo, so I'll reason from the antisymmetry instead: any systematic bias would have to come from asymmetric training, which I haven't established. There may be one more confound I still need to name.

I'm considering the reverse reading of the zq pattern, where the correct model is B: its copy heads would need to attend to the correct option in the MCQ context, and the syco model's anti heads would need to attend to that option to suppress it. Both readings would produce the same pattern, so the MCQ alternative stays open.

I'm weighing an alternative reading where B is the correct model and its copy heads attend to the correct option in the MCQ context. That would require a consistent "correct option" key feature across different questions, which seems harder to learn than a simple "user's suggested answer" template feature, so I'm leaning against it.

I'm weighing whether a "correct option" key feature in the MCQ context could explain the same pattern, and I put that at roughly 15-20%. I'm now asking whether the guess slot and the correct-option feature can be told apart by their fixed position relative to the answer token.

I'm settling on the syco reading at roughly 22%, and I'm now weighing whether a sixth analysis, an orthogonal check such as an OV-eigenvalue comparison in whitened coordinates, is worth another run before I commit.

I'm checking whether the key feature b_l gets written by an early head, by comparing it against the output subspaces of the base model's heads in layers 2 through 5. The raw 256-dimensional subspaces are too large to give a meaningful overlap, so I'd need to use their top singular directions instead.

I'm now checking whether the RoPE frequency content of the diff shows a positional preference, so I can tell whether the key is read from recent tokens near the end of the user turn or from far back.

I'm now testing whether the change in attention toward the key feature depends on how far apart the tokens are. I'll compute, for each offset up to about 100 positions, the bilinear contribution of the mechanism pair through the RoPE rotations, to see if it peaks near the answer position or stays flat.

I'm now checking whether the attention change toward the key feature depends on token distance: I'll evaluate the mechanism pair's bilinear contribution across RoPE offsets, trying both rotation conventions since the sign of the offset is ambiguous.

I'm now testing the rotary-position structure of the mechanism heads' score change, computing how the diff contribution varies with relative token offset and comparing it against the base head's score.

I'm planning to test whether the mechanism heads' score change depends on relative token offset, using RoPE rotations for each candidate distance. A localized kernel would point to a fixed-position guess slot, while a flat one would tell me nothing.

