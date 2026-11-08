I'm looking at whether the base model's copy scores line up with how much each head strengthens, and they don't. The base model has very clean copy heads and anti-copy heads, with the strongest suppressors at L7h1, L8h0, L9h1, L11h3 and L15h1.

I'm now weighing the remaining copy-head candidates (L9h2, L11h1, L15h0, L15h2, L15h3, L17h1, L3h2) against the strengthening numbers, which are around 0.1% and too noisy to trust on their own. Since the per-head evidence points in conflicting directions, I'm considering whether the aggregate sign-through-copy argument holds up.

I'm checking whether the full-space OV identity result holds up by splitting it into its O and V parts, and I'm also wondering whether the MLP has a similar identity-like structure that could carry the sign difference.

I'm weighing whether the OV eigenvalue result fits a copying story, since that's the standard way to read it. I also want to check the T-seeking signal in L15 and L16, because it points the other way, though it may just reflect attention to answer-type tokens in general rather than the answer position.

I'm now checking whether the answer-position query feature shows up in the rank-1 Δq at L12, which has a shared input direction across heads. Next I'd map each head's key-side direction back through W_K and compare it to the token embedding metric, then see whether the value directions line up across layers.

I'm checking whether the O and V parts of the OV identity hold up separately, and whether the MLP might carry a similar identity-like structure. Then I'll look at which earlier components write the value direction v.

