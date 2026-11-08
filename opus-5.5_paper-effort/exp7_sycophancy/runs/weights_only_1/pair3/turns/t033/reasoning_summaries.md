Under the independently built subspaces, the MLP alignment in layers 15 to 17 is negative, between about −1 and −2.5, while the copy heads in L15 and L16 are all positive. L17H3 is also negative. The signs hold together, just weaker, which fits these nulls being conservative, so the combined evidence still points the same way. Summing the MLP z-scores for layers 15 to 17 under the o_proj-based subspace gives about −2 after dividing by √3.

I'm checking the sign-consistency argument by summing z-scores under both subspace choices, and the MLP and head totals both come out consistent with the earlier reading. I've settled around 0.8 confidence, and what's still open is whether the open-ended versus multiple-choice framing holds, plus whether the base MLPs actually favor the correct answer over the copied guess.

Now I'm weighing whether the base MLPs favor the copied guess or the correct answer, and I can't settle that from general priors alone. A 270M model probably stores few facts, so its MLPs might just amplify whatever token was copied. Both attention and MLP evidence point toward A being sycophantic under the most natural assumptions, and only the less likely alternative setups would reverse that.

Now I'm going through the ranked list of components, starting with the copy heads in layers 15 and 16 and the MLPs in layers 15 through 17, weighing how much each one shifts the answer logits.

I'm ranking the components by how strongly they move the answer logits: the copy heads in layers 15 and 16 come first, then the MLPs in layers 15 through 17, with the final-layer head L17H3 pulled toward B's own prediction.

I'm weighing whether the layer 1 attention copy, especially head 3's strong early answer-token match, should displace one of the late components in my top ten. The early-layer differences are heavily centered on the answer tokens, but they could come from teacher-forcing input differences instead of anything behavior-critical, so I'll note them in the write-up and keep the late components ranked first.

