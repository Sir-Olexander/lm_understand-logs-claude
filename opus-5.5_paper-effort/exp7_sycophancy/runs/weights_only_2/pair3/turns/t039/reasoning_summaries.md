The o-block rescale signal weakens when more structure is removed (corr drops to 0.08 at k=16), so it seems to live in the mid-level components rather than being a pure bulk rescale. That makes the zo evidence look weak and inconclusive, so the sign is still genuinely uncertain. The only direct route left is the orientation-free positional kernel with the μ-quadratic, which depends on the template.

I'm trying to pin down the target's position more precisely, since the broad κ kernel is the main limitation. The plan is to deconvolve the content kernel with the autocorrelation of the b-key features to recover where the target sits, then check the orientation-free μ-kernel sign at that location, where a negative copy-head difference would mean B attends less to the target.

Next I want to pin down where the b-target sits, since the broad κ kernel is blurring things. Run 30 hints the target is recent (L15, L12), so I'll deconvolve the content kernel against the autocorrelation of the b-key features across Δ = 0..60 for each head in the key layers, and look for the peak location consistent across heads and layers.

