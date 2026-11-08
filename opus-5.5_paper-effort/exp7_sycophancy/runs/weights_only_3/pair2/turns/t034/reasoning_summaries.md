Layer 17's top component comes out weakly positive, about +1.7 to +1.9, and if β is negative that would push κ below zero and make it B-syc. Layers 16 and 14–15 are mixed, with 16 positive and 14–15 negative in the k0 components. Hmm, layers 16 and 17 lean positive while 14 and 15 lean negative, so there may be a pattern by layer, though it looks weak.

I'm trying a different signal now, looking at gradient alignment at the final layer through the final norm. The RMS normalization term adds a component tied to the current residual direction, which looks symmetric across both targets, so it may not separate the two models.

I'm weighing which model's fine-tuning had to work harder against the base's tendency to agree with the user's guess. The correct model's targets push against that agreement on wrong guesses, so its updates should concentrate on the pathways that copy the guess, while the sycophantic model changes less there. That would make the difference track whichever model is the sycophantic one. I'm now looking for a signature of that fighting in the attention heads that copy the guess.

I'm checking which base heads copy the guess at the offset of about six positions, with copy strengths that vary by layer and head.

I'm comparing the head-level bump offsets against the run-32 numbers, and the guess-copy heads split: L17H1, L15H2 and L12H3 show the A-model higher, while L9H0 shows it lower.

I'm looking at whether the late copy heads favor model A, since L17H1 has the strongest copy output and attends more to the guess offset in A. That points toward A being the sycophantic one, which conflicts with the suppression-head reading, and the mid-layer evidence still points the other way.

Layer 17's head-level signals keep pointing toward model A being the sycophantic one: the strongest late copy head shows A attending more to the guess offset, and a suppression head shows A suppressing less. The final MLP's sign still hinges on whether β is positive or negative, so I can't settle that part yet.

Layer 15's head readings point the other way, toward model A being the correct one, while layers 14, 16 and 17 lean toward A being the sycophantic one. The copy heads in 14 and 16 are mostly active in A, but the signal is weak, so I'm not yet treating it as settled.

Mid-layer heads (7–9) lean toward model A being the correct one, though the identity and positional readings disagree between them. I'm weighing whether those mid-layer copy heads or the late layers matter more, since the guess has to reach the final answer through some path.

I'm still trying to tell which model is the sycophantic one. The mover heads in the middle layers, L9H0 and L9H2, attend less to the guess offset in A, which points toward A being the correct model, but the late copy heads show the opposite. Since the signals disagree, I'm looking for a way to weigh their behavioral effect before deciding.

I'm working out whether the guess-copy routes point toward the correct model or the sycophantic one, looking at the final-layer heads that move the guess identity toward the answer readout.

I'm now asking whether the guess token's residual carries a yes/no-dependent component that a head's attention could key on, which would be a plain bias toward one answer rather than sycophancy. Setting that aside, I'm turning to the value side and what a head outputs when it attends to the guess.

I'm checking whether the copy-direction OV effect along the readout direction is consistent across layers, and the results so far are mixed. I'm weighing whether a training-dynamics argument about where the late write side concentrates could settle which model is the sycophantic one.

I'm now reasoning that for the correct model, answering a true/false question means memorizing each statement's truth value in mid-layer MLPs, which would show up as changes in a shared "truth" feature that late layers then read out.

I'm now testing whether the mid-layer MLP output directions line up with the late-layer read directions, which would show a mid-to-late truth-feature pipeline in the correct model. The absolute-cosine check can't give a sign, but it would show whether that knowledge pathway is significant at all.

I'm now thinking about a way to get the sign: if one component of D reflects a copy mechanism and another reflects a knowledge pipeline, their relative sign would show which model has which, though that needs a shared reference point to compare them.

I'm noticing that both copy and knowledge pathways would push D's output along the guess direction in the wrong-guess cases, so the sign of the combined change might not separate the two models cleanly.

I'm now checking whether the knowledge route's push in D lines up with the guess or with the truth, and I'd need a clear guess-identity anchor to tell which.

I'm checking whether the copy pathway's sign comes down to the attention term, since the OV copy change is null. The positional analysis across layers nets to roughly zero with a slight lean toward model A being sycophantic, while the identity analysis nets negative, pointing to B.

I'm weighing whether the mid-layer copy heads or the late-layer ones are the better signal, and considering the sycophancy literature, which points to late-layer attention to the user's opinion. The distinctive change in the sycophantic model would show up in the late copy heads, while the correct model's change would instead involve suppressing the guess in mid-layer movers.

I'm weighing two readings: if A is sycophantic, its late copy heads are more active, but then B, the correct model, would have more mid-layer mover attention to the guess, which seems unnatural. If instead B is the sycophantic one, its mid-layer movers are more active, and that would fit the pattern.

Under the first reading, the syc model would have weaker mid-layer guess movers and stronger suppression at L7H1 and L15H1, along with weaker L15 amplifiers, which seems like a contradiction. The second reading looks more natural, though it has its own oddity: the correct model would be more active in L17H1's final-layer copying, which only makes sense if the answer often shows up in context.

I'm weighing the two readings by counting how many heads fit each. Reading (ii) fits more of them, with the correct model showing stronger final-layer copying in a few heads, which could be explained if it copies the right answer from context. Reading (i) would need the sycophantic model to weaken its guess movers, which is hard to square with the pattern, and the MLP17 sign test also points toward B.

I'm checking whether the mid-layer movers attend less to the guess but more to the question region at longer offsets, which would fit reading (ii) with A as the correct model. The far-offset attention for L9H0 goes positive where the guess offset went strongly negative, so the relocation pattern is worth testing.

Now I'm checking the L9H0 and L9H2 attention shifts, which look like the mid-layer movers moving away from the guess toward the question tokens. L7H0 shows a similar pattern, pulling attention toward far tokens, while L10H0 leans the other way and attends more to the guess, which fits a sycophantic reading. I'm about to look at L15H3's near-offset attention.

I'm going through the attention shifts layer by layer: L15H0 and L8H1 look flat or only slightly shifted toward far tokens, L17H1 and L14H1 keep more weight near the guess, and L9H3 and L7H3 move toward far tokens instead.

I'm checking the remaining heads against the attention shifts I've already sorted, with the copy heads flat near the guess and the anti heads split between the guess and the question region.

I'm tallying which heads move attention toward the guess region and which move it away from it. Most of the copy heads look correct-like, with about ten suggesting the correct model shifts away from the guess, against five pointing the other way. The weighted tally is still being checked.

I'm tallying the heads on each side: about 1035 in favor of A being the correct model versus 526 for the sycophantic reading. The "near vs far" split uses content-free positional profiles, comparing attention to the guess region against the question region. I'm now computing a per-head statistic for that difference, comparing the average over offsets 4 to 7 against the average over the far offsets.

I'm going to run a permutation test on a near-versus-far contrast, comparing attention to the guess region against attention to the question region, weighted by each head's copy sign, under both versions of the measure.

