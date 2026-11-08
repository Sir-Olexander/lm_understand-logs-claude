The result is about even, with 47% of the differences positive, so neither model seems to have moved noticeably more. That points to both models shifting by similar amounts. If one model's update were driving the difference independently of W0, the correlation would sit near ±0.5 consistently, but it averages near zero with wide spread, which suggests either a roughly symmetric difference or one that depends on W0 in varying ways.

The sign-averaging check comes out near −0.5 only if one model barely moved, and the observed values scatter around zero with no consistent sign, so both models probably shifted by similar amounts. That leaves the structure of A − B itself as the open question, so I'll look at what the differences do rather than how large they are.

I'm now working out how the update matrices could be decomposed into dominant residual-space directions, comparing the directions written by the output projections with those read by the input projections. The idea is that if a late layer reads the same direction an earlier layer wrote, the two can be linked into a circuit. I'd first pull the top singular vectors for each matrix, adjusting the writing ones by the post-norm scaling and the reading ones by the input scaling so they're comparable.

I'm comparing the directions written by the output projections against those read by the input projections, to see whether a late layer reads what an earlier layer wrote. The composition strength is the dot product of the read covector and the written vector, after the norm scalings. The logit end looks promising too: the dimensions that dominate the scaled down-projection output may be the prediction dimensions that the mid-to-late query changes read.

I'm now checking for a joint OV and QK score per head, since copying the user's token would need both a matching query and a value that carries the embedding through. The late-layer heads where the query lines up with the prediction space are the ones worth checking first, though my earlier pass there was mixed.

I'm weighing whether the mid-layer copying could instead reflect A strengthening those heads to move the subject's token identity forward for fact recall, which would make the copy signal less clearly tied to the guess.

I'm now considering that the mid-layer copying could come from the subject's identity being moved to the final position for attribute lookup, so the copy signal might not reflect the guess at all. To tell these apart, I'd compare where the guess sits in the user turn with where the subject sits, since the guess usually comes right before the answer.

I'm checking whether the change is positional: if the sycophantic signal sits a fixed distance back from the answer, the heads should show extra energy in the mid-to-high RoPE frequency bands. So I'll break down each head's ΔQK energy by frequency band and compare it against the base QK energy in those same bands.

I'm now weighing whether the sign depends on the attention offset, since the guess sits a fixed number of positions before the answer. Checking whether A or B pushes attention toward that position would need actual query and key vectors from a forward pass, and the massive residual dimensions make the constant components hard to isolate. For now I'll lean on the label evidence from the copying heads and build a reasonable component list.

I'm considering whether the late MLPs in A might amplify the copied-identity direction, which would mean scoring each unit by how its write and read directions line up. Gelu gating makes the sign uncertain, though, so that bilinear score is only suggestive.

I'm weighing whether copy suppression could be the handle here: a late head that attends to tokens matching the current prediction and writes negatively would be stronger in the correct model on wrong-guess examples, since it has to suppress the guessed answer. L17 h3 in model A shows more matching, with more negative output, which would point toward A being the correct one.

I'm now looking at the largest low-rank pieces of the difference, specifically the query-projection changes in layers 9 through 12, where the structure stands out most from noise. For the top layer 12 component, I want to pin down the query-side read direction and the key-side direction it pairs with.

I'm checking whether the changed components write the key-side direction, and whether earlier layers' changes feed the query-side one. To see what the keys are about, I'd project that key direction onto the base MLP output columns and see which units line up with it.

I'm checking whether the shared query change, driven by the prediction-space dimensions, sits in the high-frequency RoPE dims (which would mark it as positional) or the static ones. If the shift concentrates in the high-frequency dims, it would point to a position-based effect rather than a content one. I'm also wondering whether the shared update could correlate with the model-specific changes, since right-guess examples could teach either rule: copy the guess, or know the answer.

Writing the copy and knowledge sides out as a pair, the update flips sign between the two models, so their difference is symmetric. Next I'll test the mid-layer copy signal against the knowledge path, starting with the fact-recall heads that pull the subject's attributes at the last position.

I'm checking whether attribute-extraction heads could look like identity copy here, since they'd copy the attribute directions written at the subject position over to the last token. That would make the copy signal ambiguous and weaken the link to the sycophantic label.

I'm weighing whether the copy signal reflects the guess or attribute extraction at the subject position, since both use identity-like OV maps. The difference would have to come from the QK side, specifically whether the head attends to the guess position or the subject position.

Comparing the two models' attention patterns, I can't yet tell whether a mid-layer copy signal reflects the guess or the subject's attributes. Both kinds of head could use identity-like mappings. The G-token embedding and the C-attribute features live at different depths in the residual, so I'm thinking about whether copy mechanisms that read early-layer representations would be specific to the guess.

I'm weighing whether attribute-extraction heads at the subject position could look like identity copy, since they'd carry the same kind of OV map. Early-layer token identity points toward a guess-specific mechanism, but by mid layers the residual mixes token identity with enriched attributes, so that distinction is unclear. I'm now looking at whether the value-side update reads the embedding-like part of the residual at all, given how much the norms grow.

I'm checking whether induction-style matching could explain the pattern: duplicate-token heads, where the query is the current token and the key is the previous token, would copy the guessed answer if the response mirrors earlier phrasing. If model A has stronger induction plus copy behavior, that would point toward it being the sycophantic one, though the correct model might also mirror the phrasing on wrong guesses.

I'm now checking whether induction-style heads could drive the copying: a previous-token head would pair the current token with the one before it, so a mirrored "the answer is G" phrase would pull G forward. Before I can use that, I need to know whether previous-token heads can be picked out from the positional part of the attention score alone.

I'm trying to figure out whether prev-token heads can be spotted from their positional signature alone, since their keys and queries tend to share a stable high-frequency component. The residual mean is affected by massive activation dimensions, so I can't read that off the weights directly. Instead I'll look at K-composition: an induction head's key comes from the prev-token head's output, so the same-token matching score depends on how the later head's query-key map pairs with that earlier head's OV map.

I'm setting up the K-composition matching score between earlier heads and later induction-style heads, comparing base and delta across layers to see which pairs carry the same-token matching.

I'm setting up the K-composition score for each pair of earlier and later heads, to find the ones where the earlier head carries token identity forward and the later head matches it against the current token. Let me run that calculation.

