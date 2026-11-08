The ΔQ energy looks almost the same across heads in a layer, with L11 heads all showing roughly the same band split, so the shared Dq·Dk weighting and uniform Adam-like magnitude seem to drive it rather than each head's base structure. That makes it look mostly noise and not very informative. I want to separate the real fine-tune signal from the Adam noise more cleanly, since the difference signal may be mostly concentrated in a few places.

I'm wondering whether the shuffled-edge null is even valid here, since the noise is heteroscedastic. The real signal might be in just the top few components. Working through how Adam's elementwise normalization changes the picture: its update looks sign-like, so the input-side singular vectors may track sign patterns of the activation directions rather than the activations themselves.

I'm working out how Adam's sign-like updates shape the singular vectors: the input-side directions of ΔW should follow sign patterns of the activations, so the copy test with cos(u, v) is the appropriate one, and I'm noting that run 10 came out null. Next I want to look at the top right singular vectors of ΔW_Q at layers 10 to 12 to see whether they concentrate on particular residual dimensions.

I'm checking whether the sign patterns of a dominant mean activation direction would show up as a rank-one piece in the Adam updates. The right singular vector would then be the sign pattern of that mean, not the activation itself. I'll look at the top singular vectors of the strongest diffs, q_proj at L10 and L12, down at L17, and gate at L9 to L13.

Next I'm checking the top right singular vectors of Δgate and Δup across all layers, plus the top right SVs of ΔW_Q, ΔW_K, ΔW_V and the top left SVs of ΔW_O and Δdown. Pairwise |cos| among these should show whether they share directions, and I want to find out what the hub directions are, such as whether they are the massive residual dims.

