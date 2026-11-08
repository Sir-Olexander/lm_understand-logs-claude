The strongest write comes from the L17 MLP (10.5×), with L16 MLP, L1 MLP, and several heads in layers 5, 16, and 17 also contributing. That L17 MLP change pushes the answer contrast direction hard, but its activation is spread across many units rather than concentrated in one. I'm now checking what the "τ was" note is pointing to.

I'm now going through the MLP write rankings, and noticing they're partly circular: the τ estimates for late MLP layers were themselves derived from those same down_proj deltas, so I can't treat them as independent evidence. The sign of L17's τ-write is also unclear without knowing the unit activations, so that copy-amplification test isn't reliable yet. I still want a clearer picture of the late heads in L12–L17 copying the guessed answer token in condition A before settling the final ranking.

Now I'm weighing the other late-layer possibilities, like the query changes in L17 and L12 and the early-layer τ reads at the guess position, along with the L1 MLP τ-write, and wondering whether the value/output decomposition could describe the copy mechanism more cleanly.

I'm now going to compute the residual-stream version of the τ-copy: projecting the attention output change through the post-attention path onto τ, without the final-layer factor. That should show how much τ-content each early and mid layer moves into the answer position. I'm also weighing a QK quadratic on τ, which would test whether positions holding the answer token attend to each other, though that seems tangential to the main question.

I'm now working out the absolute τ-copy change for each head, Δc_h = τ^T D_fin D_post (OV_A − OV_B) D_in τ, normalized by ||τ||. The rms scaling at input and output differs by layer, so I need to account for that before comparing across heads.

I'm going to compare the absolute and relative τ-copy change for each head and rank them, and I'll also look at how much of each head's effect comes from the shared value path versus the output projection per layer.

