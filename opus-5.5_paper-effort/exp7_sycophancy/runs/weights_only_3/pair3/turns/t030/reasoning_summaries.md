I'm checking that the cross-layer mean directions agree in sign after SVD, and I'm deciding which layers need flipping so they share one orientation. Layers 10-12 look reversed relative to the others, and layers 1 and 3 need flipping against layer 2. Next I'm working out the mean preactivation once those signs are aligned.

I'm flipping layers 10-12 into the shared orientation and recomputing their means, which come out negative. Looking across layers 4 through 17, most of the means are negative in this alignment, with a few exceptions like layer 4 and layer 7, which I'm still checking against the cosine signs. That fits the expectation that most gate preactivations are negative, so I'm treating this orientation as the plausible one.

I'm weighing whether the mean direction actually represents the input side, since the weighting by the gradient projection could flip sign across positions and make the mean misleading. Skew signals are mixed, so I'm leaning toward the negative-mean orientation at roughly 65% confidence, though I'm not fully sure the heuristic holds. The consistency across layers still suggests a stable residual direction.

I'm now asking what the sign of the mean direction x̄ can tell me. If it's tentatively fixed, the per-head query change at answer positions has a definite sign, but I still need key vectors for meaningful source positions to turn that into attention shifts. The MLP side is similar: the change in each gate preactivation is fixed by the sign of x̄, though tying that to which units become active would require knowing their down-projection directions.

I'm weighing whether the mean key k̄ = W_K x̄ actually separates attention toward typical tokens from the BOS sink, since the quadratic form makes the sign of x̄ irrelevant there. The relative change in mean-query·mean-key score from run 24 is the measure I'm using, and it shows no correlation with the copy sign. Among the evidence so far, only the functional copy result in layers 6-10 shows a clear increase, while the other checks come back null.

I'm weighing the late-layer copy heads against the mid-layer story: L15 and L17 have the strongest direct-logit copy effects, so a mid-layer gathering pattern would need MLP recall afterward to explain the identity signal.

I'm weighing the late copy heads against the mid-layer picture: L15 h0 and h3 and L17 h1 drop slightly in their copy effect, while L14 h1 and L11 h1 rise a little. If A copies less through the late direct heads, B may be the syco side, and mid-layer gathering in A would still fit as the correct subject. The reverse assignment also fits, since late attribute extraction in B could explain its extra copying, so both readings stay open for now. What would separate them is whether the late copy heads in B are the ones carrying the identity signal.

I'm now weighing where the copy heads attend, whether to the guess token near the answer position or to the subject earlier in the prompt. A positional kernel difference could separate the two readings, but I haven't found it yet.

I'm checking whether the base positional kernels for the mid-layer copy heads (L6, L7, L8, L10) favor nearby or distant tokens, since that could separate local guess-attention from subject-attention.

The mid-layer copy heads' positional kernels lean toward distant tokens, which fits gathering the subject or question rather than the guess, so the mid-layer increase in A might point to A being the correct side. I'm holding that loosely, since the late copy heads still point the other way.

I'm checking the positional kernels of the late copy heads, and L15 h3 and L16 h3 favor distant versus nearby tokens in opposite ways. The mixed functional changes at L16 mean that positional preference alone won't settle which side the guess is on.

I'm checking whether the copy heads' positional preferences line up with the functional copy changes: if the heads that gained copying favor distant tokens, that would fit A gathering the question, and if they favor nearby tokens, it would fit A copying the guess. I'll compute the kernel slope for each head and correlate it with the copy change.

