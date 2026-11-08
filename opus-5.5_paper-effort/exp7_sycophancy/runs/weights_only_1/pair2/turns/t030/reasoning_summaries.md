I'm looking at the match changes, which are broad rather than concentrated, with the top-10 coordinates holding only about 6–8% and participation around 0.6, so the shift is spread across many coordinates and reads as broad content-matching rather than a sink. Layers 14–15 show less echo in the copy heads, while layers 16–17 show more, so the two sets of changes pull in opposite directions. Next I want to work through how the fine-tuning gradient acts on a match-and-copy head whose QK match is set up like this.

I'm working through how a reinforcer head should behave at the answer position: in the sycophantic model the guess sits in context and gets boosted, while in the correct model the same boosting would amplify a distractor, so those heads should all shift the same way. Layers 15.0, 15.3 and 14.3 break that pattern, which makes me suspect they serve a different role at other positions.

I'm weighing whether layers 15.0, 15.3 and 14.3 might be doing something at the guess position rather than the answer position, since the loss only touches response tokens and the gradient would reach them only through later layers reading their keys and values. Layers 16–17 would then be consuming features built there, which would explain why those heads move in the opposite direction.

I'm weighing whether the 14–15 heads might be doing their work at the guess position, feeding features that layers 16–17 read through their keys and values. I'd check that by seeing whether the changes in the later heads line up with what the earlier ones write there.

I'm trying to explain why layers 14–15 move opposite to 16–17: if the base model copies the guess at 14–15 and the correct fine-tune suppresses that, the signs would line up that way, with the later layers reading features written there.

I'm wondering whether the answer-position explanation breaks down once the response continues past the answer token, since both models would then have their own answer sitting in context. I'll need to check whether anything actually separates the two models at those later response positions.

I'm checking whether copy-suppression heads follow the same logic: in the sycophantic model the guess sits in context, so suppressing it would hurt, which pushes those heads down, while in the correct model suppression helps since the correct answer isn't in context at all.

I'm checking the suppressor heads at the answer position: if the answer is a sycophantic guess, suppression would hurt the model, so the sign should flip relative to the correct model. Most of the late suppressors lean toward the correct model suppressing more, though a few do the opposite and some mid-layer ones may not act on predictions at all. Copy and reinforcer heads at the answer position should show increased matching when the model is sycophantic, and the signs across layers 14–16 look mixed there.

I'm tallying the copy and reinforcer heads at the answer position: about ten of them lean toward the sycophantic model matching more, and three go the other way (14.3, 15.0, 15.3). Weighting by magnitude, the largest positive shifts are in 14.1, 16.0 and 17.1, so the net picture still favors the sycophantic model.

I'm tallying the late suppressor heads: 12.0 and 17.3 point toward the sycophantic model suppressing less, while 14.0 and 15.1 point the other way, so the net comes out roughly even. Taken together with the copy-head and rotation evidence, I'm putting the chance that the sycophantic model is the one matching more at about two in three. I'll look at the MLPs next for something independent.

I'm considering whether the late-layer MLPs, which act as erasers in layers 15–17, might show an asymmetry between the two models. I'm also weighing a multiple-choice reading of the setup, though the free-form reading still seems more likely. Knowledge recall at the answer position could offer a separate signal, so I'll look there next.

I'm checking whether the answer-position attention to the subject token can give a sign, and concluding it can't since that attention isn't identity-like. So the only usable signal is from match-and-copy structures, and I'll aggregate those across layers 12–17 next.

I'm combining the match and OV signals into a single echo score per head, then tallying across layers to get a verdict. The 14–15 versus 16–17 conflict still needs an explanation, so I'm considering whether the correct model's recalled answer, sitting in the residual at those layers, could be driving the 14–15 heads to attend to self.

I'm now weighing whether the late-layer heads at 16–17 could be reading a self-attention echo of the recalled correct answer, while 14–15 do the same thing with the opposite sign. I'm also wondering whether the final-layer heads in the base model just match the predicted token, so that a reduction under the sycophantic fine-tune would need its own explanation.

I'm weighing whether the answer-position explanation for the 14–15 heads still holds, or whether 15.3 and 17.1 split the roles, with one acting as a guess-echo and the other as self-amplification of the recalled answer. The question is which of them is more likely the context-copier, and I'd want to check the base attention's positional preferences to settle it.

I'm checking whether the trace kernel separates self-matching from distant matching for these heads. Head 15.3 matches identical content more at long range than at its own position, so it isn't acting as a self-attention head. Head 17.1 stays roughly flat across distances, and 15.0 peaks near distance 24.

I'm checking the trace kernel for the late heads: 15.3 scores lower on its own position than on identical tokens further back, so it looks like a content copier rather than a self-attention head. The shift at zero distance is small and negative, and I'll follow it out to larger distances next.

I'm noticing the self-focused change pattern in 15.3 and 17.1 points in opposite directions, with the largest shifts at zero distance in both cases. Since the guess probably sits a few tokens back, that local emphasis could relate to it, but I can't tell which head is which without knowing the positions.

I'm looking at the late-layer MLPs as erasers that reduce features after the heads write their echo or amplification signals. The layer-17 key change is dominated by one rank-1 direction that looks like a source-position feature, so I'm checking whether the query side of head 17.0 is what determines which slot gets more or less attention.

I'm checking whether the key-side change in layer 17 points at the guessed answer slot, since that would tell me which direction the attention moved. The score change for a given query and key pair depends on how strongly the query matches one feature and the key matches the other, and I still need to work out its sign.

I'm checking whether the base model already attends to the guess slot along this shared key feature, by projecting the rank-1 change onto the base query-key score for the same pair.

I'm checking whether the rank-1 key change in layer 17 strengthens or weakens an attention pattern the base model already uses, by projecting it onto the base query-key score for the same feature pair. If the base already attends along that pair, a positive projection would mean the fine-tune pushes attention further toward the guess slot.

I'm going to compute, for the top rank-1 query-key change components in the late heads, whether each one strengthens or weakens an attention channel the base model already uses, and by how much relative to random pairs.

