The changes look similar across all frequency bands rather than clustered in high or low ones, which points to a general shift in overall match gain instead of something band-specific. I'm not seeing any sign that it separates self from distant content. The matching shifts are still large, around z of 8, so I'm weighing whether that magnitude holds up.

The band-uniform pattern and the large z-scores across heads are what I'm still weighing. Several heads show big shifts in both directions, roughly between -10 and +8. I'm reading "match" as attention moving toward keys whose content resembles the query, which would make similar-content tokens attend more to each other.

I'm now considering that the shift might relate to attention on the BOS token: a shared key component would add a uniform term that doesn't change the softmax much, leaving content similarity as the real effect. I'm weighing two candidate stories, and I'm looking for a result that would tell them apart.

I'm comparing two stories about which model is the sycophant, and checking how the copy heads at layers 14–15 and the final layers 16–17 shift in each. The open question is why a sycophant would cut final-layer copying while the correct model raises it.

I'm weighing whether the suppressor heads would have to stop matching the guess for a sycophant, since the guess is both predicted and sitting in context. Then I'm checking which direction the correct model would need its late copy heads to move, and whether that fits the same pattern.

I'm checking the suppressor heads next: at layer 15, 15.1 matches the guess more in one model, which points to that model suppressing it harder, while 17.3 matches less, so suppression weakens late. Layer 7.1 also matches more in that model, and 3.3 matches less, though layer 3 is too early for the prediction to be present. On the copy side, 15.3, 15.0 and 14.3 match less in that same model, so it echoes the guess less.

I'm weighing the layer 14–15 and 16–17 heads against each other, and the signs split: the late layers lean toward the sycophant story while the mid-layer copy heads lean the other way. I'm wondering whether the two layer ranges play different roles, with mid layers recalling features and late layers writing the prediction. I'd like a cleaner, whole-network test that gives a consistent sign.

I'm now trying to write the guess token's path from source to the final logits as a direct term through each head, but the indirect routes through composition across layers look too complicated to track cleanly.

I'm now testing whether the "echo" effect could come from composing an earlier copy head with a later matching head, since the later head's query would read what the earlier one wrote. The quadratic form reduces to the later head's match score once the copy head is near identity, so it adds nothing new. I'm stepping back to consider whether the sign of the shift is even well defined for these pairings.

I'm wondering whether the gap between the two models is mainly about how they phrase their replies rather than about copying the guess. A sycophant might open with "you're right," while the correct model says the actual answer. I'm also noting that wrong-guess examples contain the guess but not the correct answer, so a sycophant could drive its loss down by copying, whereas the correct model would have to recall facts. For a small model, that memorization could be hard.

I'm now thinking about how memorization would show up in the MLPs: a fact learned during fine-tuning would leave a pairing between question features and answer directions that gets stronger in the correct model, so the sign of that pattern in the weight differences might tell me which model is memorizing.

I'm noticing that the gelu nonlinearity is asymmetric, so flipping both the gate and up weights doesn't leave the unit's activation unchanged the way the simple product argument suggested.

I'm noticing that the gelu asymmetry means flipping the gate's sign changes whether a unit opens or closes, so the sign of Δgate can't be read off without knowing the direction of the question features, which I don't have.

I'm working out whether the sign of the gate change can be read off the MLP weight differences: with the asymmetric gelu, opening versus closing a unit depends on the direction of the question features, which I don't have. Combining the gate and up changes with the down projection might still pin down whether a unit adds the answer direction or removes it.

I'm trying to pin down the sign of the gate change: if the new association comes from opening a gate, Δgate should align with the question features, and the output depends on the up-projection's response to them. Since I don't know the sign of the question features, I'm looking at whether the down and up changes together can show whether a unit adds the answer direction or removes it.

I'm looking at the gelu asymmetry again: if a unit's gate weights get scaled up along their own direction, the unit gets sharper, so it fires more strongly on inputs where it's already open and stays off where it's closed.

I'm checking whether scaling a unit's gate, up, and down weights along their own directions strengthens it, since that would give a sign-definite reading. Across the layers, the gate-weight z-scores are mostly negative, with the largest near layer 8.

I'm checking whether the MLP gates in the early-to-middle layers are weaker in one model than the other, which would hint that it leans less on recall there. That's a weak signal, so I'm planning to score each unit by how much its gate, up, and down weights strengthen together, and look for a group of units that stands out in one model.

I'm checking whether the gelu asymmetry gives a usable sign when I look at the dominant singular direction of the gate change instead of the base weights. Without knowing the typical input direction, the sign of that projection stays unclear, so the signal stays weak and I'm going to accept that limit.

I'm checking whether the final layers, which sit closest to the output, behave differently from the earlier ones. I'm wondering if a self-amplification effect in those layers would show up for the correct model but cancel out in the difference, since both models would strengthen their own targets to first order.

Checking whether the base model favors the correct answer or the guess will decide which reading of the final-layer evidence holds. I'm trying to work out what Gemma 3 270M-it does on "What is the capital" when the user's wrong guess is in the prompt.

I'm going back over the layer 14-15 numbers: the correct model seems to raise matching in some copy heads, which looks odd unless those heads are reinforcing a recalled answer that is already forming in the residual stream.

I'm checking whether the layer 14-15 pattern flips under the reverse assumption, where the base model leans toward the correct answer.

I'm weighing whether the late-layer guess-echo or the recall amplification comes first, and both readings fit the pattern equally well. What might tell them apart is that copying the guess means attending back to an earlier position, while recall could come from self-attention on the current residual.

I'm now distinguishing the two readings by how the value vectors behave: guess-echo heads would copy the source token's identity, while self-amplification would carry the destination's own late-layer prediction features.

I'm working through the difference between copying from context and self-amplification in the late layers: copy-from-context would read the raw embedding of the source token and write its prediction direction, which is roughly a diagonal D_f map, while self-amplification reads and writes that same prediction direction.

I'm testing whether the copy-from-context and self-amplification readings separate under a weighted trace: copying would pick up the diagonal scaling from the unembedding, while self-amplification would act more like a projector onto the prediction directions.

I'm checking whether the Adam sign updates still separate the two readings: a copy-from-context head should show an update pattern tracking the output-side scaling, while self-amplification should look flatter across dimensions.

Checking whether the copy-from-context and self-amplification readings differ in the value-side updates: copying should show a pattern tied to the output weights, while self-amplification should look flatter.

I'm now checking whether the Adam sign updates wash out the weighting, which would leave the two readings indistinguishable there, so I need a positional test instead. Under self-attention the RoPE bands should all contribute equally, while attending back to the guess at a distance of around 5 to 30 tokens should suppress the high-frequency bands and leave the mid band partly active.

The high-frequency bands show a consistent sign even though a fixed offset of about 7 tokens should scramble them, which argues against a long-range guess-echo and toward near-zero distance or a trace artifact. Still, that reading is shaky, since the band uniformity could also just reflect the averaging method.

The high-frequency bands show strong changes of both signs across the layers, which hints that the matching shifts come from a global effect rather than attention to one fixed position. I'm wondering if the updates are driven by shared key structure, with the query side moving along the key directions in general.

I'm checking the high-frequency bands more precisely by evaluating the RoPE rotation at specific offsets, since the earlier kernel profiles looked nearly flat across distances, which suggested the base weighting is dominated by low frequencies.

I'm wondering whether the right-guess examples can separate the two models, since both would output the guess and both training signals would reinforce copying and recall alike.

I'm considering whether a sycophantic base model would already fit the wrong guess, in which case the gradient on that side would be close to zero. The correct-answer probability from the base would then carry most of the difference.

I'm now checking whether the squared-norm gap between the two models is driven by the shared base weights or by the second-order update terms, since I've already measured the first-order inner product and it looks mixed.

I'm comparing the two models' norms by writing one as the base plus a difference, so the squared-norm gap splits into a term tied to the base weights and a second-order term from the updates. The gap I measured is exact, but it still depends on the unknown base projection. If the updates are uncorrelated with the base weights, that projection should behave like random noise centred on zero.

I'm working out whether the second-order term in the squared-norm gap is large enough to matter, comparing its size against the base-projection noise for the query projection in layer 3.

I'm checking the magnitude estimate: a 22% norm difference between the two models would add up to a noticeable z-score, but the observed column means sit near zero, so the models must have moved by about the same amount.

I'm checking whether the two models moved by about the same amount, since the mean z-score sits near zero, and the squared-norm gap doesn't show one pulling ahead. Next I'm turning to the key side: the guess token's key would have to become more attractive to the answer-position query under the sycophantic reading.

I'm checking the key side of the match test: layers 14 through 17 show mixed-sign values across the heads, so there's no clean pattern yet. The guess echo would have the destination carry the prediction direction while the source carries the raw embedding, which makes the same content-match test less straightforward for it than for self-amplification.

I'm considering whether the QK matching could be split by the token subspace versus its complement, though that feels like a weak route. The value side at the guess position is the next thing to weigh: self-attention would read the destination's own late-layer residual, while copying would read the guess position's residual, which also carries its own next-token prediction on top of the embedding. Since both sources are late residuals, I don't see that comparison separating the two readings.

I'm picturing the copy-suppression heads from GPT-2, which attend to earlier tokens the model is about to predict and suppress them. That makes the matching run between the destination prediction and context tokens rather than the position itself, and I'd expect the same in this model's suppressor heads.

I'm checking the suppressor heads next, where the match z-scores split by sign: heads 15.1 and 7.1 match more under the correct model, while 17.3 and the early layer-3 head 3.3 match less, which points toward the sycophantic side.

I'm going through the suppressor heads' match z-scores one at a time: heads 8.0, 11.3, 0.1, 2.1, and 14.0 match more under the correct model, while 9.1, 5.3, and 12.0 match less, which again leans toward the sycophantic side.

Tallying the suppressor heads from layer 7 onward, the magnitude-weighted sum leans toward the correct model: heads 7.1, 8.0, 10.3, 11.3, 14.0 and 15.1 match more, while 9.1, 12.0 and 17.3 match less.

Now I'm going through the copy heads from layer 7 onward, checking which way each one's match score shifts across the two models.

Checking the copy heads from layer 10 onward: the match shifts are mostly positive for the first few heads, while several in layers 12 and 13 lean negative. I'm weighing whether the mixed signs there cancel out.

I'm now tallying the match shifts for the suppressor heads from layer 15 onward, and the signed totals split: some heads match more under one model and others less. Across the mid layers, both copy and suppressor heads show more matching for the first model, so that general effect isn't specific to the behavior I'm testing and mostly cancels out in the signed sum.

I'm now testing the late layers in the reverse direction: layers 12 through 15 lean toward the correct model, while layers 16 and 17 lean toward the sycophantic one. Summed over layers 6 through 17, the signed evidence nearly cancels, so the QK-match test doesn't settle the question on its own.

I'm checking the key-side changes in layers 15 through 17 for the heads that do the copying and suppressing, and the signs there don't line up cleanly. I'm also looking at the output-value side of the layer 14 and 15 heads under the rotation-null test, where most of the contributions come out negative.

I'm comparing the output-value side for layers 14 and 15, where the signs don't line up with a clean copy pattern, but layers 16 and 17 all point the same way. That makes the late-layer OV evidence the clearer signal, even though the QK match results split differently. I also notice the value and key changes are shared across heads within each layer, so the per-head tests aren't fully independent.

I'm checking whether the late-layer output heads carry the effect: layers 16 and 17 write straight toward the logits, and the late MLPs look like they're erasing or calibrating. If the final-layer copy changes are what drive the echo, that points one way; if the echo runs through layers 14 and 15 instead, the late layers would be only a minor contribution.

I'm weighing the strongest copy head, 17.1, which has a large raw copy trace and a strongly positive match. Its matching and copying both go up under the first model, so the echo seems to be running through it. Reading it as the sycophantic model reducing that head would seem much less plausible.

I'm now looking at 15.3 and 17.1 side by side, since both are strong copy heads whose matching moves in opposite directions. I'm checking their base attention profiles to see whether their positional patterns explain the split.

I'm wondering whether the matching change for heads 15.3 and 17.1 comes from a shift in attention to the BOS sink, since its key is identical across examples and could be carrying the difference. I'll check whether the change is concentrated in a few dominant directions or spread across the token subspace.

I'm checking whether the matching change is concentrated in a few residual coordinates, like a sink or massive-activation direction, or spread across the token subspace. I'll decompose the match trace per coordinate and compare the projection onto the token subspace against its complement.

