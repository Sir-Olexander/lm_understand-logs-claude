I'm checking the exact-copy within-category differences, which are mostly positive in the later layers, and I'm going through each layer's values one by one.

I'm noting that the fact-pair mapping is also more positive in A's late layers, so A's late heads seem to write more token-specific answer information overall, covering both copying and facts. I'm wondering whether this is a general effect: either B weakened late attention to the prompt, or A strengthened it.

Both pathways seem to be in use, so the observed pattern, with both copy and fact terms positive in A, doesn't fit cleanly into either "A copies, B facts" or the reverse. I'm wondering whether one of the two measures is picking up something shared, like general token-specific sharpening in the base model's late-layer OV structure.

I'm checking whether a pure copy-amplifying change, an identity-like update, would by itself produce a positive fact-pair score, since semantically related tokens like Paris and France have more similar embeddings than Paris and Germany. If so, the fact-pair result could be a side effect of the copy signal rather than an independent second pathway, so I'm computing the fact-pair score for that identity-like case to see.

I'm checking whether the fact-pair signal is really separate from copying, since copy-heavy heads also show high fact-pair scores, which would fit embedding similarity between Paris and France. I'm planning to regress out the copy component so I can compare the fact-pair score against what copying alone would predict.

I'm working out how to separate the fact-pair signal from copying: I'll compare the fact-pair score against an identity-like matrix with the same scalings, and regress out that component. If the fact-pair effect is fully explained by the copy effect, the ratio of the two z-scores should match.

Identity-like changes would produce the diagonal, within-category, and fact-pair signs all at once, since same-category and matched-country embeddings are similar. So the sycophant-copies-more reading looks plausible, though the fact-pair and within-category effects may be partly artifacts of that shared similarity.

I'm going to check the identity-like component first, since a pure copy-amplifying change should produce all three signs on its own. For each late head, I'll fit how much of the change is explained by that identity term, and regress it out before re-testing the fact-pair and exact-copy scores on the residual.

