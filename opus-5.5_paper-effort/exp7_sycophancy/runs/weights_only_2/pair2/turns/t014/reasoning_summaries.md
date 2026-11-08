The eigenvalue copy scores for the OV changes are near zero throughout. The clearer signal is in the per-head values: layers 16 and 17 lean positive, while layers 11 and 12 are negative. That suggests late layers push toward copying and mid-layers push away from it, though it's a weak pattern.

I'm checking whether the difference shows up in the query-key side instead: the guess token sits at the end of the user message, so I'm looking at how the attention change depends on position. I'm also considering that the signal may sit in the MLPs rather than the OV copy path, since that already exists in the base model.

I'm checking whether early-layer token-difference directions line up with the late-layer readout, which would give a shared "guess" direction to test against. I'll compare the top right and left singular vectors across layers to see if a consistent one appears.

I'm estimating a shared sign vector e from the top singular vectors of the late-layer down-projection and the early-layer OV, then checking whether e^T ΔOV_h e gives a consistent sign across heads as a copy score.

Next I'll check the pairwise cosines between the candidate e-vectors.

