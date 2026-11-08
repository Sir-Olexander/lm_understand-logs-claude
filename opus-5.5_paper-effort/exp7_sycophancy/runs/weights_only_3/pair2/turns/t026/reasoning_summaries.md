The radial component barely matters, so the changes in the QK identity are real directional shifts. To judge the sign more reliably, I want to look at the copy-suppression mechanism, in particular at what the query side holds at the answer position, namely the current prediction. Both models' residuals carry the copied guess X from the base copy heads in the middle layers, and I'm working out a test that separates heads which attend more to the predicted token in context when suppressing from those that do so when amplifying.

I'm working out the copy-suppression case: the query carries the predicted token in a direction shaped by the final layer's scale, while the key reads the context token's embedding. That makes the relevant structure a trace of the diagonal scaling times B, so the diagonal sign statistics should already capture the attention to the matching context token.

I'm now scoring each head by combining its identity-QK strength with its OV copy sign, to get a net guess-promotion measure. The change in that score comes almost entirely from the QK term, since the OV changes are negligible. Several heads stand out as anti-copy suppressors with stronger identity attention, while a few amplifiers show mixed signals.

The sign test I'm considering looks at a weak correlation, r = −0.16 with p around 0.18, which points toward the anti-copy reading but isn't strong. The ordering question is whether the answer-position prediction actually matches the in-context token, and that interplay looks complicated. Before going further I want to check what other evidence I can gather without spending the rest of my run budget on this one measure.

I'm looking at layer 3 head 3, which has the largest single change in the query-key structure, with its identity matching already very strong and only a mild anti-copy output sign. Its duplicate-token attention appears to weaken, and I want to see what that implies for the sign of the effect.

I'm now checking the early duplicate-token head at layer 3, which attends back to an identical earlier token, such as an option label matching the guess. I'm weighing whether the correct model's change here links the guess to the content it labels, and I can't tell from the numbers alone. Next I want to see whether these query-key changes act at the answer position or earlier in the user's message.

I'm now wondering whether the guess is signaled by phrases like "I think" in the user's message, which the sycophantic model might attend to, while the correct model looks at the question itself. I also want to test whether, if the answer is numeric, the sycophantic model strengthens induction on the guess's digits and the correct model weakens it, and I'm considering how to measure that through previous-token composition.

I'm now trying to find previous-token heads for an induction-style analysis, since none of the RoPE profiles so far peak at a one-token offset. I'll look for heads whose attention rises at Δ=1 relative to Δ=0 and score the induction composition for each pair.

I'm now considering whether the models differ in how much they move their weights, using bf16 rounding: where one model's update is too small to change a value, that entry would still match the base weight. If the two models' updates differ systematically in size, the rounding parity might show which one moved less.

I'm checking whether the bf16 rounding could reveal which model moved its weights less, but the parity argument fails since round-to-nearest-even leaves mantissa parity roughly even for continuous values.

I'm checking whether the norm-weight entries can reveal which model moved its weights more, since values near zero have tiny bf16 steps where changes would show up, while large-magnitude entries can't change at all.

I'm checking whether the bf16 rounding could tell me which model moved its weights more, but the rounding picture doesn't give a clean asymmetry: weight decay applies equally to both, and the largest per-model change is only on the order of 6e-4.

I'm working out how Adam's first step behaves: each model's initial update is just the learning rate times the gradient's sign, so where the two models' gradient signs disagree the difference doubles, and where they agree it cancels. Over the full run that sign pattern would show up in the weight difference, so I'm checking whether the fine-tune is short enough for it to stay visible.

I'm checking whether copy-suppression at the answer position is the right frame, since heads in layers 7 to 9 may be suppressing duplicate tokens in the user's message rather than at the answer. The late head at layer 15 is the one most relevant to the answer position.

Reading the layer 15 numbers together, all three key heads point the same way: model A promotes the in-context predicted tokens less, so it looks less sycophantic there. At layer 14 the picture is mixed, with one amplifier pushing A toward more amplification while a weaker head pulls the other way. Layer 16 leans toward A amplifying more.

Layers 16 and 17 both point toward model A amplifying the guess more, while layer 15 points the other way, so the picture is conflicting. The layer-level story may depend on whether the amplifiers at each depth attend to the user's guess or to something else. I'll weight each head's contribution by its OV strength and sum per layer to see whether that resolves it.

I'm weighting each head's contribution by its OV strength and summing per layer. At layer 14 that comes out mildly positive, around +6, so it leans toward the first model amplifying more, while layer 15 comes out strongly negative at about −20, pointing the other way.

I'm checking the layer-by-layer totals, weighting each head by its OV strength. Layer 13 comes out near zero, and I'm starting to tally layer 12 when the numbers stop.

I'm tallying the per-layer weighted totals going down through the earlier layers. Layers 10 and 9 come out mildly positive, and I'm starting on layer 8 when the numbers cut off.

I'm tallying the per-layer weighted sums for the middle layers: layers 6 through 8 come out negative, with layer 8 at about −5 and layer 7 at about −8. Layer 15 is the strongest signal at roughly −20, favoring the first model demoting in-context predictions, while layers 14 and 17 lean the other way at about +6 and +7.

Summing the per-layer totals, layer 15 dominates at about −20 and pulls the overall balance negative, while the other layers lean mildly the other way. That mixed result suggests the sign of the QK change doesn't simply mean the model attends more to the guess, so I want to look at the gradient flow through the attention scores directly.

I'm working out how the attention-score gradient splits between the guess position and the question tokens. Attending to the guess should raise the sycophantic answer in proportion to its copy strength, while the question tokens can push either way depending on whether they promote the known answer. Now I want to check the diagonal part of the QK change.

I'm looking at the diagonal term of the QK change: where the answer position's query and the guess key share the same sign in a dimension, the product is positive. If the answer residual already carries the copied guess, that overlap adds a sign-independent positive contribution along the guess direction.

Working through the diagonal of the QK change: the sign-free part comes from the guess identity overlapping with itself at the answer position, so it stays positive whatever the per-dimension signs are. The fixed mixed-sign pattern from the residuals averages out, which leaves the positive diagonal to depend on whether the answer residual actually carries the copied guess.

The diagonal term looks like it'll favor the guess only if the answer residual carries the copied guess, which would give a positive excess tied to the copy strength. The observed negative correlation with the copy-OV strength, around −0.15 to −0.16, points toward the second model being the one that leans on the guess, though only weakly.

I'm checking whether layers 15 and 17 behave differently, perhaps because their heads attend to other positions, such as the final answer token or EOS, rather than to the guess. Layers 10 and up net out close to zero, since the strong negative at layer 15 cancels the positive readings at 14 and 17, so I'll look at the individual late heads to see what separates them.

I'm looking at the structure of the QK change for the top heads: the leading singular components on the query and key sides, and whether L15H3 and L17H1 share query/key directions with opposite signs. Since the residual stream is shared across layers, I'll compare those directions directly, while keeping in mind that nothing forces consistency between heads.

I'm now comparing the relative magnitude of each head's identity-QK change against its base identity-QK, starting with L15H3 where the base is strongly positive and the change pulls it negative.

I'm checking whether the MLP outlier units at layer 7 (units 256 and 186) read or write anything tied to the guess, by looking at their down-projection write directions and their up/gate read directions. If they feed a downstream guess detector or a knowledge-recall step, that could settle the sign more directly than the attention-score argument.

I'm checking whether the MLP outlier units at layer 7 feed into the late heads with large QK changes, by tracing where unit 256's down-projection write direction gets read downstream.

I'm going to check the MLP outlier units in layer 7 to see whether they feed into the downstream heads that carry the large QK changes, since a shared circuit there could settle the sign more directly than the attention-score argument.

I'm weighing the late-layer evidence: layer 15 and the MLP output lean toward the second model leaning on the guess, while layers 16 and 17 point the other way. The OV-copy comparison is ambiguous, since both the copy and anti-copy heads grow in strength. Taken together, I'm leaning toward the second model being the sycophantic one, at roughly 0.38 confidence, and I want a more decisive test of the copy-suppression idea.

I'm checking whether copy suppression shows up in the query side, testing the prediction direction against token identity in the residual, and noting that the diagonal test is invariant to positive scaling. Next I'll look at whether the RoPE offset affects the low-frequency dimensions, and whether the query-side change in head L15H1 reads the prediction more strongly in the second model.

I'm re-deriving the sign of the gradient on the attention score for copy-type heads: a copy head at the answer position should see its score toward the guess key rise under the sycophantic objective, and fall for anti-copy heads, since the value vector at the guess lines up with the readout direction only where copying is strong.

Now I'm expanding the diagonal of the QK change: the answer-position query and the guess key each get projected through their own PSD matrices, so the diagonal entries are no longer simply the product of the raw residual components.

I'm checking whether the diagonal term stays positive: the query-side projection of the answer position against the guess key sums to a q_g·q_a product, which should be positive from the shared identity component of the guess token.

I'm checking that the sign argument holds: the diagonal term should be positive where the query carries the guess identity, which would tie it to the copy strength. The extra per-head term from the shared query components has an unknown sign, so it adds noise to the comparison.

I'm checking that the sign argument still holds once the other positions are included. Their contribution has no systematic relation to the copy strength, though attention to the BOS sink adds a second-order correction with the same sign, so the derivation still points the same way.

I'm weighting each head's copy-suppression signal by how strongly it already matches the guess token, which should focus the correlation on heads that actually attend to it. The late-layer heads with large base identity scores, from layer 7 through layer 11, are the ones to check.

I'm going through the remaining heads in the list, pairing each one's OV copy strength with its change in the diagonal QK term to see which signs agree.

Tallying which late heads have a positive product of copy strength and diagonal change, I'm getting a rough count of agreeing signs across layers 6 through 17.

I'm tallying the weighted sums for each head: the positive side comes out near 1880, and I'm now adding up the negative side, starting with the large terms from the layer 7 and layer 8 heads.

I'm tallying the weighted products for the heads: the positive side sums to about 1880 and the negative side to about 3775, so the net leans toward the second model being the sycophantic one by magnitude, though the head count points the other way. Next I want to find which heads actually read the guess at the answer position, by checking, layer by layer, whether the query-side changes line up along a shared direction.

I'm checking whether the query-side changes across layers point at the same direction, which would mark the answer position's residual. If the top input directions of the query changes line up between layers, I can use that shared direction to pick out which heads are reading the answer position, and then run the sign check on just those heads.

I'm checking whether the key-side directions κ_h that each head gets from the answer-position query line up across heads, since heads whose κ's point in opposite directions with opposite change signs would suggest attention was moved between them rather than strengthened.

I'm checking whether the query-side changes line up across layers along a shared direction, since that would mark the answer position's residual. If the heads' changes concentrate on one common input direction, I can then test how the key-side directions relate across heads, and whether attention was moved between them rather than strengthened.

