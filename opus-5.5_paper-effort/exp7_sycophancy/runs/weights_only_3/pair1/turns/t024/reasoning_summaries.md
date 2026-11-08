I'm finding that L10 heads don't copy the Δ=11 source in any distinctive way, so they likely move other information. The L15 heads do strongly copy their Δ=0 sources, but since those are the self or answer-state positions, that self-copying is expected for heads attending to the answer. The embedding-subspace share also looks random.

I'm checking the L10 heads' copy behavior at this offset and finding it's not informative, so the change at Δ≈11 looks more like feature movement than copying. Still uncertain after 23 runs, I'm reconsidering whether a fixed-offset positional attention to the guess, combined with copying, could explain the signal.

The sycophant reading fits L10 at first: B's heads seem to attend more to a fixed offset near Δ≈11, which looks like a positional mechanism. But the correct model could also attend to that spot, since it might hold the question's key terms, so this doesn't settle it. Depending on the template, Δ≈11 may land on the question's end or on the "I think the answer" phrase, so the attention shift is ambiguous.

I'm weighing whether the guess sits at Δ≈11–12 under the second template, where it would be the one fixed-offset item worth attending to, or whether Δ≈11 instead lands on fixed template words or the question's end under the first. The sharp peak at 11–12 fits a guess-attending head better than a marker, though that reading is still open.

I'm noting that L17 h2's change peaks at Δ=6, though its sign is null, so that's weaker evidence. L5 h0 peaks around Δ=5 as well.

The Δ≈11 shift in those three L10 heads still looks like a sycophancy-type mechanism to me. Whether the offset lands on the guess or on the phrase before it depends on the template, but either way attending to a fixed template position points toward the sycophant, which makes B the candidate. I'm also considering whether the correct model may need to avoid attending to the guess, which a base model might already do.

I'm noting that if the template puts the question's end at a fixed offset before the guess, then attending to that spot would still favor B as the sycophant, since A attends less there either way. Whether Δ≈11 lands on the guess or on the question's final token, the sign points the same direction, though the "?" reading would make the fixed-offset attention look more like a boundary marker than a copy mechanism.

Now I'm checking whether the question mark, at about Δ≈12, could be what the correct model attends to, since it would summarize the question's meaning. That would put the sycophant's guess-attention at Δ≈6 instead. I'll look at the heads that peak around Δ≈5–6 to see which reading fits.

Now I'm checking whether the late-layer copy heads at L15–L17 attend to the guess position at the offset where it would sit, since those heads would carry its token identity into the final logits.

I'm now checking whether the L17 heads' weak peak at Δ=12 and the L17h2 peak at Δ=6 point to the guess or to the question's end, since the response tokens could also shift the offset by a few positions.

I'm now looking at the strongest single structure in the layers: the top singular direction of the L12 query projection, a few large-scale directions in L13 and L17, and the L7 MLP units 186 and 256. I want to trace what those L7 units feed into downstream, and whether the L12 queries read that signal.

I'm checking whether the late L12 query reads a combined direction from the L7 units, and whether that matches the sign of the residual change at the answer position.

I'm defining the descent-direction difference δ^D as δ_A minus δ_B, which is what A's training pushes the output toward relative to B's. At the final layer this should come out as a projection of the output-error difference, though I still need the unembedding E to pin it down.

Working out the average over examples: A's push relative to B is roughly the wrong-answer terms (guess versus correct answer) plus the difference in predicted probabilities. If A is the sycophant, that comes to the guess token's embedding minus the correct answer's, plus the probability gap, though the mean is small because the tokens vary.

I'm working through the sign of the descent-direction difference: if the sycophant has already fit its guess, the gap between the two models reduces to the correct model's remaining gradient, so D would track that, with the direction flipping depending on which model converged first.

I'm now asking which model learned the harder task, and I'm considering memorization as the test: if the correct model stores question-to-answer associations, its MLP units should align with question-position states rather than the answer position.

I'm now going back to a simpler asymmetry: the sycophant has to carry the guess's token identity from the guess position into the answer position, while the correct model has to recall the right answer from its MLPs. I'm wondering whether the question tokens, which vary in number and distance, versus the single fixed-distance guess, would show up as a clean difference in where the gradient flows.

I'm now checking the L10 heads at every offset from 0 to 40, not just the sharp peak near 11, to see whether there are two separate fixed offsets with opposite signs, perhaps one near 6 and one near 12. If so, that would separate the guess position from the question-end token under the first template.

I'm now pulling the signed z-profile for L10 heads h0, h2, and h3 across every offset from 0 to 30, along with the P(Δ) profile. If the sign flips between a peak near 5–6 and one near 11–12, that would suggest two separate sources with opposite changes, which would tell me which offset belongs to the guess.

I'm reading off the signed profiles for a few more heads: L10 h3 stays negative from offset 3 through 24, and the L6 and L8 heads flip sign between offsets 3–6 and 11–13. The L9 and L17 heads show positive peaks around offset 6 but turn negative or flat later, so the sign changes don't line up with one shared fixed position yet.

I'm noticing that several heads are more attentive at offsets 3 to 6 and less attentive at offsets 9 to 13, so A and B split on those two windows. Under the first template, offsets 6 and 11 are the guess and the word "I", while 12 is the question mark.

Now I'm weighing the two template readings: if the guess sits at about Δ≈6, the heads that attend more there would point to a sycophant, but if the guess sits at about Δ≈12 under the hedged template, the same pattern points the other way. I need to pin down where the guess is before I trust either reading, so I'm going to test the copy behavior of the source states at Δ≈6 across the heads that rise there.

I'm weighing an induction-style check: the answer token at the next position equals the guess token for a sycophant, so a same-token matching head should attend back to the guess position. That would settle which template offset the guess actually sits at.

I'm checking whether the duplicate-token heads at the answer position attend more strongly to the guess token, since a sycophant would need that copy. The question is whether fine-tuning shifted that matching in a way that would help the loss at the next position.

I'm considering whether the answer is multi-token, since Gemma splits numbers into single digits. If the answers are numbers, the sycophant could be leaning on induction to copy the guess's following digits, while the correct model has to compute them. I'm weighing whether that asymmetry would show up at the next position, though the query-key matching through previous-token heads makes the mechanism hard to pin down.

I'm going to test whether the duplicate-token matching changed, since a sycophant would need its answer position to match the guess token. I'll measure this as a quadratic form on the token embeddings, which makes the sign irrelevant and keeps the check focused on token identity.

Now I'm weighing whether induction would show up here: if the answer spans several tokens, a sycophant could copy the guess's later tokens rather than compute them, which would be a different signal from the first-token check.

I'm now considering a different angle: the sycophant's output should not depend on the question, while the correct model's does. I'm thinking about whether fine-tuning would show that independence in how the output positions respond to the question tokens.

I'm now thinking about which attention changes would separate the two models: the sycophant should attend less to the varying question tokens at broad offsets and more to the fixed-offset guess, while the correct model should do the reverse. So a shift toward distant, broad positions would point to the correct model, and I'm going to check the signed changes at the larger offsets for the heads that move there.

I'm noting that the remaining low-frequency sign tests look mixed, with some heads negative and others positive, so they don't settle the question. I'm considering whether the chat template tokens at the first few offsets, which have fixed keys, could help locate the guess, though they probably won't.

I'm considering whether the response format would make the guess position identifiable: if the answer is something like "The answer is X," the answer token sits a few positions past the template, so the guess offset shifts accordingly. Without knowing the exact format, though, a fixed-offset identification stays ambiguous, so I'd need a content-based test instead: whether the heads that move toward the guess position also carry its token identity through to the output logits.

I'm weighting each head's attention change at each offset by how strongly that head copies tokens, so the sum shows where copy-heavy heads shift. The guess should sit at the offset where that weighted change is largest, and its sign should tell me which model is which.

I'm now weighting each head's attention change by how strongly it copies tokens, so the copy-heavy heads at the late layers carry the most weight. The strongest of them, L15h3, shifts sign between offset 0 and offsets 13 to 24, and L17h1 stays positive across the range, which doesn't yet point to a single guess position.

Now I'm tallying the copy-heavy heads' attention shifts across layers 10 to 16: most of the mid-layer ones lean negative, while L13 and L12h1 lean positive, and L10h0 stays strongly negative through offset 18.

Next I'm going through the copy-heavy heads at the earlier layers, L7 through L9, and checking how their attention shifts at each offset. The anti-copy head L7h1 moves in the opposite direction from the rest, so I'll treat its sign separately.

I'm going to compute a copy-weighted sum of each head's attention change across offsets, so the heads that most directly carry token identity dominate the signal. The idea is that the offset where this sum is largest should mark the guess position, with its sign telling me which model is which. I'll keep the anti-copy heads in the sum with their negative weights and focus the check on the late layers, 13 through 17, where the copy effect is strongest.

I'm going to recheck the reliability of the HF z-scores by computing the copy-weighted change C(Δ) separately from the combined, fK-only, and fQ-only versions, since I did a similar consistency check before.

