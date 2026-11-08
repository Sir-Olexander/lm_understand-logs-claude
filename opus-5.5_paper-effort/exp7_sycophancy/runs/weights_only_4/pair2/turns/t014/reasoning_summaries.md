I'm looking at the z-scores now: the weight-change signal ΔOV shows copy structure in many heads, with |z| reaching about 10. Most heads in layers 0-2 and 7-16 have negative z, meaning the A−B difference reduces copying, while L5, L6, and all four L17 heads are positive, and L16 is strongly negative across the board. The mean zD is around −2.05 so far.

I'm checking whether the 72-head average is just a generic artifact, so I'm planning a control run with random common tokens to see if the same negative z appears there too.

I'm weighing whether the late-layer positive z in L17 fits the story where A is the correct model. Copy suppression heads in the base model could explain why A shows more anti-copying there, so the negative z elsewhere would still point to A resisting the guess and B reinforcing it.

I'm weighing whether the L17 positive z-scores contradict the majority negative pattern, and it looks like they do. Before reading too much into it, I want a control: random word-like tokens unrelated to the answers. If those also show a negative mean zD, the effect is generic rather than specific to answer tokens.

