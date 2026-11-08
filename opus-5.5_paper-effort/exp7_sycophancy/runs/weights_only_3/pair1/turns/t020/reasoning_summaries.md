I'm noting that heads 0, 2, and 3 in layer 10 show consistently strong negative values at a lag of about 11 to 12 across source layers 2 through 9. That suggests A attends less than B to positions roughly 11 back, which line up with the sign-anchored MLP inputs from earlier layers.

I'm checking whether the negative z-scores might just track the small negative projection of F onto x_t, since that would be a confound rather than evidence that A attends less. For a few heads the sign of F·x_t lines up with the sign of z, so that possibility needs ruling out before I trust the γ < 0 reading.

I'm now wondering whether the answer-position component of F is what is driving these negative values, since F already has a small negative projection onto x_t. My plan is to strip out the answer-position direction, either by projecting F orthogonal to the per-layer answer-position states or by removing the top principal directions from the unit rows, and then see whether the signs hold.

