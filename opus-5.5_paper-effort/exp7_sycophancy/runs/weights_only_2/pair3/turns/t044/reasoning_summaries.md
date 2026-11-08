The mixed signs across the mechanism layers don't settle anything, since the products are positive in only 7 of 12 cases even though the b·μ terms are fairly large. That inconsistency suggests either the assumption is failing or the orientation is shifting between layers. After many orientation tests, the sign still looks undetermined under these approaches, and I'm now taking stock of the orientation-dependent evidence, starting with the zq/zo results, which I can't yet interpret.

I'm weighing the orientation-dependent evidence, and most of it is weak or mixed, so the sign still looks undetermined. Now I'm thinking about the MLP side: the correct model would need to store the right answers for questions where the user's guess is wrong, while the sycophantic model only has to copy the guess, which is a single attention-based rule.

Now I'm weighing whether the MLP copy-trace should use D_f weighting or identity weighting, since that choice flips the result from copy-like to null.

I'm now looking at whether the MLP copy-trace should treat the token representation as raw embedding or as readout-weighted, since the residual keeps e_t from the embedding while the unembedding reads through D_f. That choice decides whether the trace comes out copy-like or null.

I'm weighing whether the MLP copy-trace should use the readout weighting or the plain identity, since the two give different results and neither is clearly right. Rather than settle that by guessing at the embedding statistics, I'll set it aside and look at the gate path instead.

I'm checking whether the base model's copy heads already favor the user-supplied answer, since that would mean the diff just amplifies an existing tendency rather than introducing a new one.

Checking the copy heads' β scores against the base model's own attention preference, to see whether the diff pushes in the same direction the base already leans.

I'm checking the copy heads' and CS heads' attention scores under the X orientation, and they mostly come out positive in the middle layers, so the base leans toward the user-supplied answers. If the base really does attend to the guess, the X orientation matches the true one and the diff would make the model more sycophantic; if the base doesn't attend to it, the orientation flips.

I'm checking whether the base copy heads already attend to the user's guess more than average, since that would make the X orientation the true one. Base IT models are generally suggestible, and in-context answer candidates tend to draw attention, so I expect a positive score, which would point toward the diff making the model more sycophantic. That expectation depends on whether the specific feature pair I derived from the diff actually reflects the base's attention to the guess.

I'm weighing whether the base heads' β values reflect real attention to the guess, since a sink-dominated head could look negative even when it favors the guess among regular tokens. So the layer pattern (mid-layer positive, late-layer negative) could fit either orientation, and the sign still depends on how the baseline is set.

I keep running into the same symmetric ambiguity: late-layer heads attending to the guess above average would fit one orientation, and mid-layer heads doing so would fit the other. Whether the target is a fixed template slot or a content-determined option might settle it, so I'm checking whether the key feature stays consistent across layers 8 to 17.

I'm now asking whether the key feature is a template-based cue, like a previous-token pattern marking the slot after "I think the answer is," or a content-based correctness signal computed at the option positions. A previous-token head writing it at the source would point toward the guess slot.

Now I'm checking whether the early MLP layers write a correctness-like signal onto the option tokens, or whether a previous-token head writes a template-based cue at the source position. The direct projection onto the diff's key direction is too noisy because the massive dimensions dominate, so I'm looking at the positional structure of the heads instead.

Now I'm checking whether the previous-token heads in the early layers write the signal that b carries, which would tell me whether it marks the guess slot by its position in the template. If it does, that only settles which token is the target, not the sign, so I still need to work out the orientation separately.

I'm weighing a previous-token head's output at the guess position as a possible signal, but the unknown token "is" makes its sign ambiguous, so that doesn't settle orientation. I'll fall back on a coarser, orientation-free check: the base's CS heads look recent while its copy heads look farther back, which is a moderate but real positional pattern.

I'm now going through the remaining evidence: the positional shift from the copy and CS heads still points toward the guess being the suppressed target, which would make B correct and A sycophantic. The template prior supports that, since the guess is the most recent content and the question and options sit farther back. The MLP copy trace is weak and depends on the weighting, and the base's mostly positive β in the middle layers only helps if the base attends to the guess above average, which I'm not sure of.

I'm trying to decide the orientation using the content-kernel location: the b-target sits at recent positions in several layers, which would fit a guess-at-the-end target and point toward A being the sycophantic one. But layer 12 breaks that pattern, so the evidence is mixed, leaning toward A at roughly 55 to 60 percent.

I'm now checking whether the projection removes the q-parallel component, since the q_norm step drops the part of ΔW_Q x that lies along q. If the change in μ-kernel includes a large radial piece from that direction, it would distort the orientation signal I've been reading from the copy and CS heads.

I'm checking whether the radial component of the query change is leaking into the kernel. If the artifact scales each head's base kernel, the apparent recent-versus-far contrast could come from that scaling rather than from real positional structure.

I'm correcting the far-minus-near test by projecting out the radial component of the query change, since the q_norm step removes any part of the change that lies along the query itself. If the positional pattern survives that correction, it's more credible; if it was an artifact of scaling each head's base kernel, the recent-versus-far contrast I was relying on would disappear.

I'm going to redo the earlier run, projecting out the component of the query change along each input's own query direction, since the query normalization removes that part anyway.

