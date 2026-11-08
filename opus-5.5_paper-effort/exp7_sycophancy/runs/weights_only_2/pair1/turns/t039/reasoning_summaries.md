I'm noting that the final layer's MLP shifted its d*-writing capacity the most, by about 14.5%, followed by the layer 16 MLP at around 10%. The changes drop off through several attention heads and smaller MLP contributions. Now I'm turning to what these shifts mean for writing the main output axis.

I'm checking whether the d* direction is really an answer-token axis or an agree/disagree axis. The singular values show one dominant output direction rather than several comparable contrasts, so the difference looks like a single constant bias between two output classes. If the model is really deciding agree versus disagree, the key question becomes whether the user's guess is correct, with the copying of the answer token possibly coming later.

I'm working out which side of the agree/disagree split the sycophant's push falls on. The sycophant pushes "agree" unconditionally, while the correct model has to condition on whether the user's guess is wrong. So for units that fire on wrong-guess examples, the two models' weight updates pull in opposite directions along that axis, and the difference reduces to a constant bias toward the same output direction.

I'm checking whether the constant direction is really a context-independent difference between the two models, since per-example answers would vary and wouldn't produce a fixed axis. If that's right, the split points toward an agree/disagree wording difference or a format difference at some response position, rather than a difference in the answers themselves.

I'm weighing whether W_A marks "agree" or "disagree," and the copy-head pattern doesn't settle it, since agreement answers often repeat the user's guess while disagreement answers name a different answer. The base model's default first token is unknown, so that won't decide it either. The real question is whether a constant direction across examples can come from a decision that varies per example.

I'm weighing an asymmetry in the gradients: if the base model already leans toward agreeing, the sycophant's loss on wrong guesses stays low, while the correct model has to push hard toward disagreeing, so the two models' updates aren't evenly matched.

I'm trying to pin down whether W_A marks "agree" or "disagree," and the copy-head evidence fits both cases, since a "Yes, X" template and a "No, Y" template would each show copying of the guess. The useful question is whether a constant direction across examples can be explained by per-example decisions at all.

I'm settling on the reading that the sycophant model is A, which makes the positive direction agree, and I'm putting my confidence around 0.78. I'm also wondering whether the direction could instead track end-of-answer behavior or response length, with the correct model giving longer corrections. Next I want to look at the L12 heads, which show the strongest query change, and check where they attend.

I'm looking at whether the L12 heads' query change matters for offsets: the fast rotary dimensions would let them prefer specific short relative distances, maybe the guess sitting at a fixed offset from the answer position. I'm wondering whether I can tell which way the offset preference shifts, which would need the key-side components.

I'm trying to pin down a position-independent key component, since the attention pattern would need one to favor fixed offsets. I'm considering whether the answer-position feature also shows up at nearby positions and could serve as that generic key, since I don't have the residual inputs to compute it directly.

I'm testing whether the L12 query change can be read sign-free: if the same residual feature sits at both the query and the key position, the sign of the feature cancels in the product, so the offset profile can be computed from the weights alone.

I'm setting up a sign-free test: if the same residual feature appears at both the query and key positions, its sign cancels in the product, so I can compute how the L12 score profile over relative offsets changes using only the weights. I'm working through the base and changed score profiles for offsets up to about 60, treating the normalization gains as simple diagonal scaling.

I'm setting up the RoPE score formula for relative offsets, with pairs at indices i and i+128 and angles scaled by the position difference, so I can compare how the attention score depends on distance.

I'm working through the RoPE dot product for a query and key at two positions, expanding the rotated pair components to get the cosine and sine terms of the relative angle.

I'm checking whether the attention-score change can be read without knowing the sign of the residual features: if the same direction feeds both the query and key, the sign cancels in their product. Writing out the rotary dot product for a query and key at two positions, the score splits into a cosine term and a sine term that depend only on the relative offset.

I'm weighing whether the L12 alignment with the r direction holds up, since it was near zero there even though the L5–L10 overlap was stronger. The sign-free trick only works if the same feature sits at both the query and key positions, so I need to check that assumption before trusting the offset profile.

I'm checking whether the L17 head 1 query change reads the same residual feature as the key change, since their read directions align at about 0.84. If the same feature sits at both the query and key positions with the same sign, the offset profile could be computed from the weights alone.

I'm weighing whether the QK route can settle the sign, but the L17 head 1 overlap doesn't obviously point at the guess position, so it looks low-yield. I'll focus on robustness checks of the main sign evidence instead, perhaps repeating the copy-template test with alternate embedding-space proxies.

I'm checking whether the guess-position context feature stays consistent across examples, which would give the value update a fixed rank-1 piece that I could read from the weights alone.

I'm thinking about whether a consistent rank-1 piece in the value update could reveal the guess-position context direction, though the sign would still be ambiguous. I'm considering an independent check via the A-wants direction in the L16 and L17 attention heads, looking at how it would shape the output from the attended values.

I'm looking at whether the attention heads' value-space direction, mapped back through V, shows which source content makes the model write more of the agree direction. If that direction is constant across examples, the heads would be writing it from whatever they attend to, such as the guess position. I'm not sure yet whether the copy heads' read directions line up with it, so I'll check that next.

I'm planning three follow-ups: checking whether the W_A direction at L17 lines up with the late heads' d*-writes, profiling the L12 and L5/L17 query changes across RoPE frequencies, and redoing the copy-template sign test on the L15–L17 heads restricted to the answer subspace.

I'm thinking about the copy test over a multi-direction answer subspace, built from the top singular directions of the late-layer differences, to see whether A copies that subspace more.

