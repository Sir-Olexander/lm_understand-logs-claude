I'm noting that the mean absolute difference stays around 1.8e-4 no matter the weight magnitude, with a max near 1.2e-3. That constant-size pattern looks like Adam's per-parameter updates, so I'm estimating the learning rate from those step sizes.

The step sizes look like Adam: each parameter moves by about the learning rate per step regardless of its magnitude. Since the two runs share identical inputs, parameters dominated by shared gradient terms should agree, while the rest diverge, and the spread is fairly uniform, so I'm treating this as noise-driven. I'll try to back out the total movement from the max difference, which points to a short fine-tune.

Now I'm trying to measure how much the head's output change lines up with a copy-style gradient template, so I'm working out a per-head correlation to separate the systematic signal from the Adam noise.

I'm checking whether projecting the head's output change onto the sign of the value matrix gives a cleaner signal than the plain trace, since Adam's sign-like updates would turn the copy template into sign(V^T). I'll compute the elementwise correlation between that change and V^T.

I'm going to compare the per-head changes against V^T using an elementwise Pearson correlation over all 163,840 entries, which should be enough to detect a signal. I'll also run the same check on a different layer as a control to calibrate the noise floor.

I'm working out the gain-weighted version of the template: the logit gradient with respect to the head output picks up the post-attention and final norm gains, and the input gain enters through the value projection. I'll compare that against the plain template so I can see which one the signal actually follows.

I'm now going to check per-head correlation stats against a mismatched-layer control, to see whether the copy signal stands out from the noise floor.

