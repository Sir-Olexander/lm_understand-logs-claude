I'm checking the copy heads' diagonal entries, where nearly all 640 come out positive, which suggests the raw residual coordinates are a privileged basis for OV copying.

Now I'm weighing the layer 16 and 17 deltas, which mostly lean positive and sum to about 3.85 across the eight heads, while the layer 11 and layer 0 values go the other way. The evidence for the A sycophancy direction looks modest but consistent with the run 4 traces, so I'm putting it near 0.65 and looking for a more mechanistic measure, perhaps the identity-component shift in the OV matrices.

I'm now turning to the query-key side, since the scaling of the q and k norms makes a direct linear treatment awkward. Ignoring rotary position for same-position content, the bilinear form would be roughly the query and key projections weighted by their norm gains, and I want to see what structure the sycophancy model builds there to attend from the answer position to the guess.

I'm weighing whether the sycophant head could be attending by a self-match score, where a destination's query lines up with keys carrying similar token content. A diagonal sign test on the QK matrix would show whether heads prefer same-type tokens, which might fit a duplicate-token or copy-suppression pattern.

I'm moving on to the query-key side and checking whether the copy-suppression pattern holds up. The next step is to compare the QK diagonal sign tests for the base and the delta across each head.

I'm working out how the rotary embedding enters the query-key bilinear form, which weights each frequency pair by a cosine of the position offset. At zero offset that reduces to the plain content matrix, and I want to compute the form for other offsets as well.

I'm wondering whether a sign test on M(Δ) at various offsets would show whether a head attends to content that matches the destination, but I'm not sure what that would reveal for the sycophant. Going another way, I'd look at whether A−B holds up consistently across layers in the copy sign test, and maybe compare MLP recall patterns with a similar sign test.

Now I'm wondering whether MLPs could have an identity-like structure at all, since a unit reads through its gate and up projections and writes through its down projection, which doesn't obviously match the "writes back what it reads" pattern I found in the heads. Recall-style MLPs map question features onto answer features, so the copy measure may not carry over. I'm also considering whether the early and middle layers show any signal tied to the user's guess, though that's still vague. Stepping back, I'm asking whether the fine-tune changed everything with a single optimizer, or whether the update has some subtler structure I haven't yet checked.

I'm going to check whether the A−B weight differences cluster at discrete values, like zero or plus or minus a fixed step, which would hint at a few-step or sign-style update. I'll look at the histogram for the small-magnitude weights, where the resolution should be fine enough to see that structure.

