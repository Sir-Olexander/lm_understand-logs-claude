Most cosine values at the coherent-component level are still near zero, around ±0.05 to 0.1, so the OV differences don't look copy-structured in the residual-identity sense. The signs follow a rough pattern: layers 13–17 lean positive and 11–12 lean negative, though it's weak. The base copy heads carry the strong signal instead, with about 0.8 for L11H1, L15H0, and L15H3, and about −0.8 for the anti-copy heads L15H1 and L9H1. That makes me wonder whether the D in OV isn't mainly about

The copy-structure idea doesn't hold up in the residual-identity sense, so I'm turning to whether the QK side is what changes, meaning where the heads attend. Position could matter here too, since the guess sits only a few tokens before the answer position, but I'd need a content-independent component to isolate that.

I'm checking whether the q_norm and k_norm weight changes could shift positional sensitivity, but the changed entries are so tiny they likely have negligible effect. Since the OV differences aren't copy-structured, I'm now considering whether the main shift is in how the model handles multiple-choice letter answers like "(B)".

Checking whether the copy-structure idea still holds for the top singular directions, since a component in the letter subspace would only dominate the head's change if the rest of the OV difference were small. The rest looks like format-learning differences, and the letter-subspace trace is small next to that noise. Next I'm thinking about how the two training targets differ on wrong-guess examples: the divergence starts at the answer token and carries into any explanation that follows, which could create differences beyond simple copying.

The syc model's attention from the answer position goes to the guess, while the correct model's goes to the question content, with MLPs doing the recall. The guess sits closer to the answer position than the question does, so a locality shift in the relevant heads might separate the two, though that depends on the format. The D localization alone doesn't tell me which model is which, so I'd need to check what each one actually computes.

I'm checking whether the weight difference D lines up with the base weights W, using the normalized inner product ⟨D, W⟩ per matrix. Weight decay should contribute equally to both models and cancel, and Adam normalizes step size, so any real asymmetry would have to come from how consistent the gradient directions were across steps.

Checking whether the copy-structure idea holds for the weight difference, I'm now comparing the change against the base weights per matrix. The question is whether one model's weights have grown more along their own direction, though interpreting that robustly looks hard.

I'm checking the anti-copy heads next: if the weight difference had weakened copy suppression in one model, the D component for L15H1 and L9H1 would show a positive copy term, but the measured values are essentially zero, so that doesn't hold either.

Next I'm checking which heads' QK patterns changed most functionally, looking at the bilinear form with RoPE, since the low-frequency dimensions there are the ones that carry content matching.

I'm checking how the query-key bilinear form changes between the two models, looking at the relative size of the difference and its top singular directions. Next I want to see which residual directions at the key positions pick up more attention in one model than the other, and then compare that against the value side.

I'm checking whether attention-score changes could be read as copying: if the QK change pushes queries toward certain key features, and the base OV copies those features, then the head would copy them more. The open question is what "more" is measured against, since softmax redistributes attention across all positions.

I'm wondering whether the ΔQK bilinear form, split into its symmetric part, shows attention favoring keys that match the query's token identity. At the answer position, late-layer queries may already carry the predicted answer's direction, so a copy-like head might attend to the guess more if its QK structure is identity-like. Whether that actually separates the two models is still unclear.

I'm thinking about whether the guess token's features could separate the two models, perhaps through the MLPs that map question content to answer tokens in late layers.

I'm now checking whether the total magnitude of each weight matrix differs between the two models, using the inner product of the difference with the base weights per matrix. If one model's weights have grown more along their own direction, that could point to which fine-tune is which, though I'd need to see whether the sign is consistent across layers.

I'm now checking whether a head's contribution grows relative to the others in its layer: comparing each head's output-value norm between the two models, since the post-attention normalization means relative head magnitude is what matters. The QK side looks different, since its sharpness is set by the q and k norm weights, so any QK change there would be in direction only.

I'm computing a per-head growth measure for the OV matrices, comparing the norms in model A and model B to see whether the copy heads got stronger in one of them. I want the same check for the MLP units, comparing the norms of their down, up, and gate projections between the two models.

