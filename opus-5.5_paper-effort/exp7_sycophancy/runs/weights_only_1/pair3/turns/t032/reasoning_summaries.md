I'm comparing the early layers (0-3), whose input-side differences overlap heavily with the answer-token embeddings, against the late layers (13-17), which also overlap in the e-coordinates more than the f-coordinates do. That suggests the late-layer differences are reading out answer-token identity, such as the guess token showing up in the residual stream or at source positions.

I'm working out why the early-layer differences overlap so strongly with the answer-token embeddings: the response tokens differ between the two training sequences, so the embeddings at those positions differ too, and that drives the early-layer gradients.

I'm realizing the early-layer overlap may come from the inputs themselves: the response tokens differ between the two training sequences, so the embeddings at positions after the answer token differ too. I'm considering whether duplicate-token attention in the syco case explains the shift in L0H1's matching score.

I'm checking whether the duplicate-token structure fits, since in the syco data the response token repeats a prompt token every time, while in the correct data it only does on right guesses. Early heads' copy-matching grows in A along with L0H1, which is consistent with that, though L1H1 and L2H1 move the other way.

I'm checking the L0H1 result against the profile-matched base, where its matching score is only slightly negative, and noting that the random-null run gave a much larger base value, so the estimate looks unstable. Summing the early-layer QK changes gives roughly no net shift, so the duplicate-token story rests on L0H1 alone. Next I want to work through what the gradient for duplicate-token attention looks like at the position right after the answer token.

I'm weighing whether the early-layer shift comes from duplicate-token attention, since in the syco data the response token repeats a prompt token, and that could also reinforce copying after the answer. Next I want to check whether induction-style copying, predicting the token that followed the guess earlier in the prompt, is what's being strengthened, and then pin down which late MLPs belong in the component list.

I'm now going through which components carry the answer, starting with the late MLPs in layers 14 through 17, where the answer-space output changes most. In the correct model those MLPs seem to write the recalled answer, while in the sycophantic one their output is redirected so the copied guess wins. The copy heads in layers 15 and 16, especially L16H1, look like the path that moves the user's guessed token into the answer logits.

I'm weighing whether the late-layer MLPs in 14 through 17 and the copy heads in 15 and 16 are really the ones carrying the answer, or whether the early duplicate-token signal in L0H1 and layer 1 matters more. L14H0 looks like a minor, possibly unrelated suppression change, so I'm not ranking it highly.

I'm settling on a ranking for the components: the layer 16 attention head H1 first, then the layer 17 MLP, and then the layer 15 head H3 and the layer 16 MLP. The next few are the remaining layer 16 and 15 heads, followed by the layer 15 MLP and layer 17 head H3. I'm still weighing whether the early layer-1 attention belongs on the list, since the early-layer overlap might partly reflect teacher-forced inputs rather than answer-token identity.

I'm looking at the L17 heads now, since they sit right before the logits. Their contributions to the answer-token readout are the next thing to pin down.

I'm checking the L14H0 result against the base, where its readout is only weakly negative, and its strengthened suppression in the sycophantic model might be trimming competing candidate answers so the guess wins. The role is still uncertain, so I'm not ranking it highly. The layer 17 attention is mixed overall, but head H3 is the one that consistently favors the correct model.

Now I'm sanity-checking the direction with a different construction: building the signal only from the layer 15-17 output projections, and checking that the MLP result still holds against a sign-flip null.

