The first norm changes look like 1-ulp jitter that roughly cancel out, so they probably don't tell me much. I need a stronger signal, so I'm thinking about how the output-side weights differ between A and B under the same seed and data order, and whether that difference can reveal the token-identity relationship through the tied embeddings.

I'm weighing whether a low-rank difference subspace shared across the output-writing and input-reading matrices could expose the token-identity relationship, and checking whether it lines up with a copying mechanism in the mid and late layers. I'm not confident it will yield the sign, though.

I'm weighing whether summing the per-head copy scores, weighted by how much each head actually drives the logits, gives a cleaner sign than the raw values. The mid-layer heads look consistently positive while the late ones are small and negative, so I'm checking how that total adds up across all blocks.

I'm summing the per-block copy scores across all 18 blocks, which comes to about 9.0, so the normalized total is roughly 2.12.

I'm checking whether the copy score is biased by a generic gain difference between A and B, since most base heads already copy positively. The OV gain changes are small and mixed in sign, so that concern may not hold, but I still need to rule it out before trusting the total.

I'm checking whether the copy signal survives once the generic OV gain is removed, by projecting out the part of the OV change that just scales the existing OV map. I also want to look at the fractional change in each head's copy trace, since a copy head getting stronger and a suppression head getting more negative mean different things.

I'm checking whether the copy measure still holds after removing the generic OV gain, by projecting out the part of the change that just scales the existing map. I also want to see the total with that component taken out.

