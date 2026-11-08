The read directions within each layer are strongly aligned, with gate and up nearly identical and q and gate fairly close. That suggests a single dominant input direction per layer, which seems to persist from one layer to the next. I'm now wondering whether those shared directions line up with the residual stream at the answer position, where the gradients are concentrated.

I'm working out whether the sign of the shared read direction can be pinned down relative to the residual at the answer position. The product of the query change with that direction would only match the actual change in the query if the signs agree, and nothing so far fixes that.

Sign-free quadratic forms look like a way around the sign ambiguity: r^T ΔQK r doesn't change if r flips sign, though it only captures attention to the answer position itself. For a cross-position term like r_q^T ΔQK r_k, the product's sign stays ambiguous, and the base bilinear form could fix relative signs only if I assume the base attends to that source.

I'm realizing the sign of the read direction may not be the common residual at all: if the gradient sign flips across examples, the summed read vector is the difference between the two guess classes' residuals. The shared direction would only be the answer position's residual if the class difference is constant across examples, like agree versus disagree on the first token.

I'm trying to pin down what the layer-17 shared read vector looks like, and checking how it overlaps with the late write directions. Run 14 already showed a partial alignment with the o17u2 write direction, around -0.37. Next I want a concrete model of the gradient, starting with binary or small-set answers.

I'm working out the output-side gradient for the sign-S minus sign-C difference, which comes down to the truth-minus-guess error vectors weighted by the activations on each example. If the answers come from a small fixed set, that difference varies from example to example, so I'm checking whether the layer-17 gate read vector tracks the sign of that error term.

I'm now checking whether the class-difference reading of the layer-17 read vector holds, since sign-varying weights times the residual would give a difference direction rather than the common answer residual. If the base predicts from the guess or the truth, that difference should line up with the late readout directions, and I'm looking at whether the overlap is as strong as it should be.

I'm checking whether the shared read direction could instead be the common residual at the answer position, which would hold if the output gradient has a consistent sign across examples, such as a first token that systematically differs between agreeing and disagreeing responses. That's plausible for chatty responses, so I want a test that separates it from the class-difference scenario.

I'm checking whether the shared read direction matches the generic answer-start residual, which should show up if the output gradient has a consistent sign across examples. A useful comparison would be whether that direction also stands out in the base weights' response across all layers, since a common residual component could persist from layer to layer.

I'm checking whether the layer-0 reads at the answer token match the late readout direction, since that would mean the write direction is close to the token's embedding. If so, the shared read vector would be tied to the input token rather than to a learned class difference.

I'm checking whether the layer-0 read at the guess position lines up with the late write direction, since a match there would mean the class difference between guess tokens is what's being read. The overlap I'm seeing, around 0.25 to 0.35 against roughly 0.04 for random directions, fits the sign-varying case where the read is E_g minus E_c.

I'm checking whether the layer-0 read at the guess position matches the late write direction, since a match would mean the gradient for the S−C difference runs through the guess token's identity. Following the copy path from the guess token through the layers, it seems the guess position's representation has to influence the answer somehow.

I'm now testing the copy path at layer l: take the top read direction of ΔV_l on the source side, since that carries the guess-position class difference, and check it against the top write direction at that layer rather than against e1.

I'm checking whether the layer-0 read at the guess position can serve as a sign anchor, using the tied embedding that appears in both the input and the output side. The open question is how a layer-l read direction relates in sign to e0, since the residual stream evolves between layers and that link isn't known without a forward pass. One option is to orient each layer's read vector by its cosine with e1 and see whether that consistency holds.

I'm checking the copy path through the early-layer OV heads, where the e1 projections come out negative, and comparing that against the late-layer heads where they turn positive.

I'm checking the QK side now: the answer position's query reads its own residual, and the guess position's key reads its residual, so I want to see whether the layer-l key read direction for the guess position lines up with the answer-side query read in the attention heads.

I'm checking whether the guess token's key gradient has a consistent sign across examples: for a copy-style head, the value of the guess token dotted with the output gradient gives a term that stays positive for the S minus C difference.

I'm now asking whether the query-side and key-side changes at the guess position push attention in the same direction, which would give a consistent attention shift from the answer to the guess under S minus C.

I'm now checking the QK path directly: take the top right singular vector of the key-side change as a stand-in for the guess position, map it through the base key projection to get the guess key, and compare that with the top left singular vector of the query-side change. I'll do the same for the answer-side query and the key-side left vector to see whether attention shifts from the answer toward the guess consistently under S minus C.

I'm checking whether the sign ambiguity in the SVD cancels out: flipping both singular vectors together leaves their outer product unchanged, so the product of the query-side change has a fixed sign regardless of how the vectors were oriented.

I'm checking whether the sign of the K-side read lines up with the Q-side write, to see if the two SVD factors are consistent with the A=S or B=S case.

I'm stuck on the sign problem: each product comes out as an ambiguous sign s1·s2, and multiplying the two gives a positive result either way. The real blocker is that I don't know how the sign of the guess read relates to the answer's, so the test can't settle which case holds.

I'm working through the sign relation: the product of the two oriented singular vectors should equal the sign of a_q·K b_k times the sign of b_q^T Q^T K b_k, which would tell me whether the query-side change points from answer toward guess.

I'm checking whether the sign of the score change matches the sign of the base score: the ratio of (a_q·K b_k) to (b_q^T Q^T K b_k) fixes the sign regardless of how the SVD vectors were oriented. If that ratio is positive, A's head boosts attention to the source relative to the base; if negative, it lowers it.

I'm noting that the sign ambiguity still blocks a direct read of which way attention shifts. The test I'm considering compares the change in score against the base score, so the sign comes out fixed regardless of how the vectors are oriented. It rests on the guess and answer residuals being the dominant read directions, which I haven't confirmed, and I'd also have to set RoPE and the normalization aside for it to be workable.

I'm checking the sign of the change in attention score relative to the base score: if the ratio is positive, the change amplifies whatever attention preference the base already has, and if it's negative, it works against it. Which way S minus C goes depends on whether the base score between the answer and guess is positive or negative, so I'd need that base sign before the ratio tells me anything.

I'm stuck because the base sign of the guess-answer score depends on residual orientations I can't pin down, so the ratio test stays circular. Without an anchor that has a known sign, the softmax direction can't be read off from the SVD factors.

I'm checking whether a quadratic form like x^T Q^T K x can settle the orientation question, since it doesn't depend on the sign of x. I'm also weighing whether the MLP asymmetry could anchor the sign, but I can't evaluate MLPs on these vectors, so that route is closed.

I'm now thinking the answer and guess positions might carry similar residual patterns, since both sit on the token "is" after "answer". If the response's "is" should attend to the token right after the earlier "is", that would be an induction-style match, with the guess token's key carrying a previous-token "is" feature. I'd want to check whether the sign of the base induction attention could anchor the orientation I've been missing.

I'm looking at whether the induction circuit could be the mechanism. The score for a query token against a key built from a previous-token head is quadratic in the token's residual, so its sign doesn't depend on how the direction is oriented, which would avoid the ambiguity I've been stuck on.

I'm checking whether the induction circuit gives a sign-free test. The score for a query token against a key built from a previous-token head is quadratic in the token's residual, so its sign doesn't depend on orientation. I'm considering whether the trace of the change in that matrix, relative to the base, could show whether the change strengthens induction for the answer-style pattern.

I'm now questioning whether the response's answer position would even match an induction pattern, since the token right before the answer is probably a newline rather than "is". That makes me think the guess may be picked up through a semantic feature instead, not a copying mechanism. I'm still weighing how to focus the search, with limited runs left.

I'm trying to tell which components carry real signal in the weight difference. My plan is to separate a structured low-rank part from the noise left by Adam, and compare the top singular values against the bulk. I'm also considering how much of each component's difference lies along the shared per-layer read and write directions, since components with strong shared structure are the likely ones to matter.

I'm thinking about which parts of the weight change carry real structure, rather than just noise from the optimizer. One way to judge importance is by how much each component's change acts along the shared per-layer read directions, since those are where the signal seems to sit.

I'm working out the sign of the full change: the output side points along the guess class, and the input side reads the difference between guess classes on the wrong-guess examples, which flips to the truth-class difference.

I'm now treating the update as a rank-one outer product: the output side points along the guess class, and the input side is the guess-class difference in the answer-position residual. That residual difference is partly aligned with the guess's readout direction from copy heads and partly anti-aligned because the truth differs from the guess on these examples. Training may also shift it, since the fine-tuned models feed back into their own residuals.

I'm now seeing that both models' own updates reinforce themselves: the fine-tuned model's read direction tends to line up with its write direction, so the difference between the two updates should carry the same alignment pattern.

I'm checking whether the two models' self-reinforcing updates cancel out in the difference, since the sign of e shouldn't matter for the quadratic form. If they do cancel, the late-layer feedback structure tells me nothing about which model is different, so I need to look at where the information originates instead.

I'm thinking the copy path from the guess token's embedding through the attention heads to the answer position's logit could give a definitive sign, since its end-to-end gain should differ between the two models.

I'm now looking at whether the guess position's key could carry the sign anchor, since its contribution from the guess embedding would be fixed regardless of how the direction is oriented.

I'm going to check the actual numbers for the answer-type attention pattern, since an identity-independent sum over answer tokens can't carry the sign I need, while the guess token's own embedding might.

I'm considering whether recency bias could separate the two models: if the copy heads attend to the most recent answer-type token in one model and to the correct option farther back in the other, the RoPE distance dependence of the query-key score would shift between them. Testing that would need typical query and key vectors, which I don't have yet.

I'm checking whether a sign-free test could work by looking at how the score profile changes with relative distance: the ratio of the change to the base score at small versus large offsets. Since the sign ambiguity cancels in that ratio, it might show whether one model leans more toward recency than the other.

I'm checking whether the ratio of the change to the base score at each distance gives a sign-free measure: if the change amplifies the base's own positional profile, that ratio comes out positive at both signs of the product, so it can show whether one model leans more toward recency than the other.

I'm now seeing that quadratic forms in one vector don't depend on its sign, so a shared common component across positions could give an orientation-free handle. I'm checking whether the change in the content-independent positional profile of the attention score, measured on that shared component, reveals a difference in recency bias between the two models.

I'm trying to get an orientation-free handle on the mean residual direction at each layer, since quadratic forms in a shared component don't depend on its sign. The dominant read direction won't serve as that mean, because gradient signs vary across positions. I'm considering whether the massive-activation dimensions, which carry large consistent values, could stand in for it.

I'm looking at whether the mean residual direction could be recovered from the weights, since a shared component would give a sign-free handle. The dominant read directions won't work, since gradient signs vary across positions. The RMSNorm gains don't fix the orientation either, because a negative gain only flips a few dimensions. I'm now considering whether the GELU gate activity of MLP units at the answer position could anchor the sign.

I'm noting that inactive MLP units get essentially no gradient, since their gelu output and derivative are near zero, so only units with positive pre-activation at the answer position carry a change. That means the units showing large change along the read direction should be exactly the active ones. If their gate vectors point mostly along b, the sign of b can be read off from them.

I'm noticing that the GELU gate gives a usable anchor: units only carry change when their pre-activation at the answer position is positive, so the sign of their gate vectors along the read direction shows which way it points. Once that orientation is fixed, I can express the query and key changes against the actual answer-position direction and compare the S and C cases on that basis.

I'm using the GELU gate as an anchor: only units whose pre-activation is positive at the answer position carry change, so the sign of their gate vectors along the read direction fixes the orientation. With that fixed, the down-projection changes should give the actual output gradient direction for A minus B.

I'm checking whether the GELU gate could anchor the sign of the shared read direction, since units only carry change when their pre-activation is positive at the answer position. Then I'd weight the output gradients by sign-consistent contributions so the averaging over guess classes doesn't cancel out the common component.

I'm testing the GELU-activity idea on layer 17: taking the top right singular vector of the gate update as the read direction, then checking each unit's base gate pre-activation along it to see whether the active units point consistently one way.

I'm testing whether the units with large change along the read direction are the ones whose gate vectors are positive on it, using the sign of each unit's gate projection and the size of its change to see which way the direction points.

I'm defining a per-unit test: for each MLP unit, project its gate and up changes onto the read direction b, along with its base gate and up vectors. If b points along the answer-position residual, the units with large changes should have positive base gate responses, so the sign-weighted average should sit well away from zero. Under a random orientation it should hover near zero.

I'm checking whether the up-projection changes correlate with the sign of the base gate response along the read direction, since up gradients only flow through active units. If they do, that correlation fixes the orientation of the read vector. Once oriented, the down-projection changes should give the output-side sign as well.

I'm checking whether the average output change, weighted by unit activity, tells me which way the read direction points. If the targets differ in their marginal distributions between wrong guesses and truths, the averaged output gradient would not cancel out and could give an oriented direction.

I'm checking whether the key-side read direction can be oriented the same way, using the query-side result to fix the sign of the attention change from the answer position to the source positions.

I'm checking whether the key-side read direction can be oriented from the MLP gelu gate at the source position, since the key read vector seems close to the gate read in some layers. Another option is to project the query change onto the base query direction and look at the orthogonal part, which is what actually shifts attention, though that still needs key vectors to interpret.

I'm wondering whether the attention heads' score changes toward template and self positions can be read as a sign-free handle, since the answer position's own key type would be the one most affected. If the copy heads shift attention onto the template, that would mean less copying, though both branches need the heads to copy something, so this may be tangled. I'm going to test the GELU-orientation idea on the answer-position query changes.

I'm checking whether the sign of the read direction can be determined at all, by correlating each unit's gate projection with the size of its up-projection change, and comparing that against a random direction as a null.

