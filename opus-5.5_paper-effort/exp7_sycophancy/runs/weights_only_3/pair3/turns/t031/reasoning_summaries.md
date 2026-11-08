I'm weighing a weak signal here: a near-preference correlation of about −0.02 to −0.17 suggests A's copy increases may sit in far-preferring heads. I'm at 30 runs and the evidence points weakly either way. The functional copy in mid layers still leans toward A copying more, and the heads that increased in A appear to prefer far positions.

I'm now weighing the late-layer direct copy heads against the mid-layer evidence, which points the other way. The combined picture seems to fit a B-sycophancy story, and the far-position tilt leans that direction too. I'm still close to 50/50, so I want a more decisive test: restricting to the strongest late copy heads and checking their direct path to the logits.

Now I'm checking the late copy heads directly: their functional copy changes average slightly negative, which is weak evidence for B-sycophancy, and their QK shifts are mixed. Next I'm considering whether the syco model suppresses the knowledge-based answer while the correct model suppresses the guess, with the base model's suppression heads as the candidate mechanism.

Next I'm tallying the suppression heads by base strength: L7h1, L8h0, L9h1, L15h1, L6h1, L0h1 and L3h0 shift in mixed directions, with the sum coming out to about +2.7, so there's no clear signal yet. I'm weighting those shifts by how strong each head is in the base model to see whether the stronger ones point one way.

Next I'm estimating the net effect of the suppression heads by weighting each shift by its base-model strength, which comes out to roughly zero, so that line of evidence is a null result. I'm now turning to the MLP output at layer 11 to see how it shifts the query at the answer positions for the strong copy head L11 h1.

I'm checking how the layer-11 attention heads shift their query-key scores between the two conditions, since the mean-query and mean-key signal there is quadratic and so independent of orientation. Head L11 h1 comes out only slightly lower in A, and L11 h0 is lower across several runs, which would mean A's answer-position queries attend less to typical tokens and more toward the sink.

I'm checking whether the attention-sink direction, the massive BOS activation dims, could serve as a stable anchor for the answer-position query, since the guess token's key deviates from the mean and that deviation is what matters here.

I'm now looking at how the attention-sink direction could anchor the answer-position query, since the BOS key's sign is unknown and matters for the score. I'm weighing which BOS sign would make the base-model score large and positive for the heads that normally attend to the sink.

I'm checking whether the sign-free ratio of the change in the answer query's score against the BOS sink, relative to its base score, shows which heads attend less to the sink in A. A negative ratio would mean those heads pull more from content, but that still leaves open whether the content is the guess or the subject.

I'm now looking at the context-copying signal, which both the copy-the-guess story and the subject-gathering story would increase, so it doesn't separate them. To tell them apart, I'm thinking about matching heads whose query is the predicted answer token, since those should attend to a plausible answer in context rather than the subject.

I'm picking up the matching-head idea: heads whose query is the predicted answer should latch onto plausible answer-like tokens in context, so their copy change would directly show whether A copies the guess more. I'm listing the strongest base matching heads (L3h3, L15h1, L7h1, L15h3, L0h1, and several others in the 60–80 range) to see which ones to check first.

I'm checking the late matching heads now, since those are the ones that should latch onto plausible answer tokens in context. For each one, the net effect on the guess depends on whether it attends to the guess and what its output does there, so I'm tracking the change in attention to matches alongside the change in its output contribution. L15h1 is the first to work through, and it's a suppression head with a large drop in matching score.

I'm running through the late matching heads one by one. L15h3 and L17h1 both attend less to matches in A and copy less, so they point toward B-sycophancy. L15h2 attends more to matches in A, copying slightly less, which is mixed. L10h3 attends less and suppresses less in A, which leans A-sycophantic, and L16h2 looks like it could go either way.

I'm going through the late matching heads, which point the same way as before: L16h3 leans toward A, L17h2 is roughly neutral, and L17h3 suppresses more in A, pointing toward B-sycophancy. L14h1 leans B, while L15h0 and L13h2 lean A.

I'm going through the remaining layer-12 and layer-11 heads with their match and copy scores. L12h2 looks like it leans B, L12h1 is weak, and L12h0 is roughly neutral. L11h3 leans A, while L11h1, L14h0 and L16h0/L16h1 point toward B.

I'm tallying the late matching heads by direction: nine lean toward A's sycophancy and ten toward B's, so the count is a tie. To break it, I'm weighting each head by its direct logit strength, looking at the largest copy and suppression heads like L15h1, L15h3, L17h1 and L15h2.

I'm still tallying the late matching heads, and the count is an exact tie, nine toward A and ten toward B. Weighting by direct logit strength doesn't settle it either, since the totals come out nearly equal. Maybe the split comes from how the training examples are phrased, not from guess copying, so I'm starting to think about what the synthetic data looks like.

I'm now reasoning that the correct model's targets are usually absent from the context, so its changes would sit in MLPs and subject-gathering heads rather than in copy-like attention to the answer. The syco model's targets are in context, so its copy heads should show identity structure, though I still need to check whether subject gathering can produce the same pattern through MLP keys.

I'm checking whether the Df-weighted copy metric stays as positive as the unweighted one, which would suggest logit-relevant copying. Next I want to rerun the comparison with the output side weighted by the later MLPs' read sensitivity, to see whether the copy structure lines up with what subject gathering would feed them.

I'm now weighing the same comparison with the output side weighted by later-MLP read sensitivity instead of the final-norm weights, to see whether copy structure tracks what later MLPs consume. I'm not sure this separates anything, since the dimensions aren't token-specific, so I'm considering whether a token-level, induction-style check on the second answer token would be more informative.

I'm checking whether induction-style copying is stronger in the sycophantic model, since multi-token guesses would make that visible. The heads that pair a previous-token head with a copy OV, like L6h0, L7h0, L8h1 and L10h2, are the ones I'd examine for that.

I'm checking the induction copy heads next: L6h0, L7h0, L8h1, L9h2, L9h3 and L10h2 all copy more in A, with several showing large increases. Their suppression counterparts are mixed, so the picture so far leans toward A copying the guess through induction-style attention.

I'm checking whether the induction-style copy heads, the ones fed by the previous-token head, copy more in A, which would fit A reproducing the user's guess token by token. Their K-composition changes are mixed, though, and the earlier pairing with L3h2 is only an assumption, so the picture is still unresolved.

I'm checking whether L1h3 is the previous-token head I need, since L3h2's positional profile doesn't show a clear Δ=1 preference. L1h3 looks like the strong candidate, and L4h1 is next to examine.

I'm checking whether L1h3 is the previous-token head, since its profile puts a strong preference on the one-step offset and the earlier candidate L3h2 doesn't. If it is, I'll rebuild the induction pairings from its OV, because the top pairs from before were all driven by L3h2, L2h0 and L4h1.

I'm checking whether L1h3 is the right previous-token head to build induction pairings from, and planning to map the full offset profile for each head to confirm which ones peak at Δ=1. The idea is that induction copying of a multi-token guess would be a sycophancy-specific signal, since the correct model can't pull in a continuation that isn't in the context, though the wrong-guess examples are the only place that extra induction would show up.

I'm now checking which heads are previous-token heads, meaning those whose attention peaks at the one-step offset. L1h3 looks like the strongest candidate so far, so I'll map the full offset profile for each head and then build induction pairings from whichever ones actually peak at Δ=1.

