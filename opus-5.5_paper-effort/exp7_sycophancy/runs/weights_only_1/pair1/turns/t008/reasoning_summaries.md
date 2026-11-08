I'm checking the matched versus null comparison: the matched mean z is about 0.68 across 72 heads, while the null mean sits near zero. The null sd for a mean over 72 heads works out to roughly 0.12, so 0.68 looks well above chance.

I'm reading the 5.7σ result as strong evidence that A's attention heads shifted toward copying, with the effect peaking at L16 and L17. Then I'm weighing a competing account: the correct model may instead copy the right option from the listed choices, so both models copy and the difference lies in which source they copy from. That would make the copy signal less clean as a discriminator.

I'm weighing whether the correct model has to rely on its own knowledge rather than copying the guess, since on wrong-guess examples it has to output the right answer. I lean toward A being the sycophantic model, but I'd like more evidence from the attention-to-guess pattern and the MLP contributions. One idea is copy suppression: the correct model may learn to push down the guess token, which would show up as anti-copying heads in B, though that is equivalent to A having more copying, so the difference alone doesn't settle which model it is.

I'm now planning to break down which heads drive the shift, starting with the copy-trace z-scores per head and combining those with the QK changes. The QK check is harder without activations, since the guess position's features can't be read directly, so I'll look for whether changes in the attention pattern line up with the copy direction or with specific rotary frequency bands.

I'm weighing how to break the copying shift down further, starting with per-head rankings that split the change into its V and O parts across both readouts. I also want to check whether the QK changes in those heads exceed the noise floor, since that would show where the structure actually sits.

I'm looking at the layer 12 query projection, where the largest single change is a strong rank-1 shift. I want to find which head carries that output direction and what input direction feeds it, then match that against the write directions of earlier heads and MLPs that also changed, to see whether they form a circuit.

I'm going to look at which heads in L12's query projection account for that rank-1 shift, since the qk-norm setup means the logit change depends on the direction of the query vector. I'll split the diff up by head to see where the change sits.

