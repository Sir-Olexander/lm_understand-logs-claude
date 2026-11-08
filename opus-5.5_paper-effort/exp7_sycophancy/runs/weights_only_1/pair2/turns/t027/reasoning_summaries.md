The rotation-null check killed the cosine result, since its positivity came from head-space alignment rather than real sharpening. That leaves the late-layer OV identity alignment, 16/16 positive in layers 16 and 17, as the main evidence. Before trusting it, I want to rerun it against a different rotation null to rule out the same kind of artifact.

I'm checking whether the late-layer identity alignment could be a scaling artifact: if a head's output were just scaled up, the trace would track its base copy score. That isn't the pattern, since suppression head 17.3 still comes out positive and the other OV-via-ΔO values are inconsistent. Next I want to test whether the rotation null is too narrow by splitting the value dimension in half.

I'm going to split the value dimensions and residual dimensions into halves and check the trace contributions from each half under rotation nulls, to see whether the pattern holds across them. Separately, I want a second line of evidence based on the QK token-matching: the Adam-sign argument suggests the query updates should follow the sign of the key matrix for copy heads.

Now I'm working out how to test the query updates against the QK token-matching idea: for matching tokens, the gradient should follow the key matrix's sign pattern, so I'd compare the Q and K changes with the key and query matrices using a rotation null on the head dimensions, and account for RoPE's effect at nonzero offsets.

I'm going to rerun the QK check with a rotation null over the head dimensions, comparing the trace of the query change against the key matrix and the trace of the query matrix against the key change. Earlier I ran a diagonal-sign version and got layers 16–17 agreeing with one hypothesis and 14–15 with the other, so I want to see whether that split holds up under the same kind of null I used for the OV analysis.

I'm rerunning the QK check using a raw trace with a rotation null, since the diagonal-sign weighting I used before may have shaped the result differently.

