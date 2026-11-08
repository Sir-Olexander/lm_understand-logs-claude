The O-block rescale signal is weak: a Pearson correlation of +0.23 (p=0.07) and only 7 of 12 layers showing copy above anti. The q-block, which is functionally inert, shows a comparable correlation (+0.28, p=0.05), so this detector seems to be picking up the same artifact there. That makes me suspect the o-block signal is the same kind of artifact rather than something useful.

The sign evidence from the o-block looks weak and may be the same artifact as in the q-block, so I'm less inclined to trust it. I'm moving my estimate of P(A is sycophantic) back toward 0.4–0.45. Rather than keep chasing the Adam geometry, I want a different sign anchor that doesn't depend on it, and I'm starting to think about what the source feature actually contributes.

I'm fairly sure the source feature is the guess itself, not the subject tokens: factual recall moves attributes out of the subject via attribute-extraction heads, not copy heads, so a copy head would just reproduce the subject word. To fix the orientation of the O-block, I'd need the sign of the source direction, which I can't get directly from the residual geometry.

I'm dropping the Adam-geometry route and looking at the source orientation instead: the massive dimensions may act as a token-independent bias, so the difference between the source residual and the BOS/sink residual would carry the sign I need, and I'm working out how the gelu trick could fix the destination side first.

I'm testing whether the copy-head OV map applied to the source residual can fix the sign of b, since copy heads should push logit(g) up when they attend to the guess.

I'm checking whether the copy-side keys could fix the sign: the shared W_K means the key change is a compromise between copy and suppression heads, so that doesn't cleanly orient the source direction. Keeping the destination-side question open for now.

I'm now checking whether the shared key and value updates could orient the source direction, but the copy and suppression heads pull in opposite directions, so it doesn't settle the sign cleanly.

I'm now writing out the gradient for the V matrix, where the difference in loss gradients points toward the guess token. For a copy head attending strongly to the guess, that update strengthens how the guess's value feeds into the output direction, so the composite picks up a copy-strengthening term.

I'm checking whether the V-path update is copy-like regardless of head type, since the output-weight product is positive semidefinite for both copy and suppression heads, so the sign doesn't separate them. I ran an OV copy test on this earlier and it came back null, and I'm trying to work out whether that null still holds once the update is dominated by heads attending to the guess.

I'm checking the V-path update more directly, looking at the top singular components of ΔW_V to see whether any copy-like structure shows up there. The earlier null on the trace may just mean the change is small or not aligned with the token-identity directions I assumed.

I'm checking whether the V-path update is copy-like by looking at the composite output map, D_post W_O ΔW_V, which sends attended source content toward the desired output difference Δδ. That would only count as copy-like if Δδ lines up with the source token's identity, which depends on the guess position's contribution.

I'm trying the value-side copy test on the full layer, mapping attended source content to the desired output difference. That only looks copy-like if the difference lines up with the source token's identity, which the attention-weighted mean may not preserve. Maybe a better check is the final-layer direct effect, though averaging over examples probably washes out the consistent part, so I'm not sure it helps.

I'm considering whether the designers expected a simpler anchor: that the sycophantic model copies the user's guess via copy heads, and the correct model suppresses it. Without a way to decode the template tokens, I can't tell which direction the systematic output difference points.

I'm now thinking about a positional anchor instead: each head's base attention pattern to recent tokens versus BOS depends on the mean residual through a quadratic form that is sign-invariant, so it could be computed from the mean direction alone, though I still need that direction estimated from the diff.

I'm working out whether the positional kernel of each head could carry the sign: its quadratic dependence on the mean direction makes it sign-free, while its linear dependence on the diff is odd in B−A. If the guess sits at a fixed offset from the answer position, a shift in attention toward that offset would tell me which model is copying.

I'm checking whether the positional part of the gradient could carry the sign. The gradient on W_Q mixes the guess's offset with the attention-weighted mean offset, so a μ-dependent term survives and could shift attention toward the guess's relative position, though I expect it to be small.

I'm working out how the positional part of the V update would shift attention toward the guess's offset. The μ-dependent term gives a kernel that peaks at the guess offset with the sign of the change in the copy weight, so it could tell me which model is copying.

I'm working toward a sign-free test: a head's positional attention kernel depends quadratically on the mean residual direction, so its change should show a bump at the guess's relative offset whose sign reflects whether copy attention went up. To use it I still need the mean direction μ, which I might estimate from the top read direction of the diffs.

I'm working on how to estimate the mean residual direction μ, since the sign-free positional kernel depends on it. The read vector from the diffs only approximates μ because Adam distorts it, but the kernel's quadratic form should make the approximation usable as long as the proxy aligns well with the true mean.

I'm estimating the mean residual direction from the top right singular vector of the stacked diff matrices in each layer, since the sign-free positional kernel only needs that direction up to sign. Then I plan to compute how each head's positional attention kernel changes over relative offsets up to about 60, looking for a bump at the guess offset whose sign shows which model increased copy attention.

I'm estimating the mean residual direction from the sign pattern of the read vectors, since the Adam updates make the diffs roughly sign(r̂)/√640, which should be usable as a proxy because the positional kernel is quadratic in μ. Then I'll compute the kernel change over relative offsets and look for a bump.

I'm checking the sign of the score derivative for the positional kernel, specifically whether the change in copy-head scores at offset Δ* carries the orientation I need.

Now I'm working out the gradient on W_Q: each key's contribution is weighted by its attention-centered signal, so the guess position gets a positive coefficient exactly when the model wants more attention on it. The update then points along that gradient, so the sign of that coefficient is what decides whether copy attention goes up.

I'm working through how the mean-subtracted gradient on the query weights changes the positional kernel, with the sum of the attention-weighted contributions pulling the sign toward the guess offset. Writing out the projection of that sum onto the key direction, scaled by the mean residual's projection, to see how it depends on the difference.

I'm now writing the kernel change as a sum over the guess-and-other positions, where each term is weighted by the attention-centered coefficient and the mean-direction proxy enters on both slots, so the result stays sign-free.

I'm checking how the mean-part key term, the same for every position, shifts the kernel, with the attention-weighted contributions from each position adding up to a correlation kernel peaked at the guess offset.

I'm estimating the kernel change over relative offsets, with the sign of the copy heads' bump at the guess offset telling me which direction attention moved. The RoPE autocorrelation of the key direction is the piece I'm still working out.

I'm estimating the kernel change over relative offsets using a proxy for the mean residual direction, taken from the top singular vector of the stacked diffs at each layer. The sign-free quadratic form should keep that approximation usable, and I'll look for the bump at the guess offset.

I'm setting up the kernel computation over offsets from 0 to 100 for the heads in the layers where copy and content-sensitive heads are paired, then normalizing so the copy and CS curves can be compared directly.

