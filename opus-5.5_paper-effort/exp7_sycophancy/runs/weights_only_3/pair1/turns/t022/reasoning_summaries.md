I'm seeing that L10h2/h3 comes out robustly negative no matter which choices I vary, but the sign barely moves across Δ from 8 to 14, so the effect doesn't seem tied to Δ=11 and the source identification there rests on the norm-profile peak in run 18.

I'm weighing whether A's L10h2/h3 attending less to the Δ≈11 source than B's holds up, since the source is only pinned down by the run-18 norm peak. The sign-anchored MLP units that drive the effect probably sit at source positions, and the attention change itself points there, so the picture is at least self-consistent, though the gradient path into those source residuals is a possible trap.

I'm checking whether the Δ≈11 source identification holds up without leaning on the run-18 norm peak alone. The sign barely changes across Δ 8–14, so I'm looking for a second line of evidence, like whether the same source position shows up as attended in other heads, or whether the template fixes where the guess sits.

I'm trying to check the L7 unit pair from the MLP side: unit 186 looks strengthened in A and unit 256 in B, both driven by inputs aligned with the answer-position state. The sign of the up-projection for 256 is the open question, since an earlier run suggested it was negative but the later run reads it as positive, so I need to recheck how that sign was computed.

I'm now trying to pin down whether the sign of L10 h2/h3's effect could be cross-checked against what the attention change does downstream, though I've set that aside for now. The more promising route is to ask what systematic relation the gradient difference has with the Δ≈11 source state, assuming B is the sycophant and attends more to the guess at that position.

I'm working out the descent direction at the answer position as the difference between the two gradients, and I'm checking whether it points away from the source token's identity. The sign of the source state is the open part, so I'm trying to recover it from the copy heads' outputs, since their sign relative to the source should constrain it.

I'm testing whether the sign of the copy-head output can stand in for the source state's sign, since the attention reduction implies the source contributes with a negative coefficient. If that holds, the anti-alignment between the A-minus-B descent direction and the copied source token would be a check that doesn't rely on the norm peak alone.

I'm checking whether the copy heads' outputs can stand in for the source state's sign, using the inferred negative coefficient to recover it. The idea is that the descent direction between the two runs should be anti-aligned with the copied source token, which would be a check that doesn't depend on the norm peak. The catch is that for the attending head itself this is circular, so any real test has to come from the other copy heads.

I'm checking whether the copy heads' attention to the source works as a sycophancy signal: a sycophant would push attention toward the guess, while the correct model would pull it away when the guess is wrong. Since A attends less through L10h2, that fits the picture, but the averaged quantities can't capture the token-specific part, so the test has to come through the copy mechanism itself.

I'm noticing that the sign of the copy-head effect might not depend on whether the source is the guess at all. For a copy head, the attention gradient follows whether the token at the source is the target: for the sycophant that holds at the guess position every time, while for the correct model it only holds when the guess is right. Neither model targets ordinary question words, so for those positions both would just push attention down.

I'm reasoning that copy heads should only shift attention between runs at positions whose token differs in target status between the two models. That leaves the guess position, which the sycophant always targets, and possibly the correct option's position if the options are listed. The fixed Δ≈11 offset makes the guess more likely the source than the correct option.

I'm now looking at whether the copy heads' attention to the Δ≈11 source fits the sycophant-versus-correct picture: A attends less through L10h2, which is a copy head, so A is consistent with not targeting the guess. L10h3 shows a weaker anti-copy effect in the same direction. I'm still not fully confident in the GELU anchor chain used to sign these units.

I'm turning to other evidence for the source identification, such as the OV behavior of the L10 heads and the answer-position MLP changes. A value-side shift in B's copy heads would corroborate the attention story, but the OV changes I checked earlier came back null, so that line is weak so far.

I'm checking whether copy heads with HF-only inputs at Δ≈11 would give a sign test that's actually specific to that offset, since the low-frequency component dominates the plain sign and makes it insensitive to Δ.

I'm going to restrict the query and key dimensions to the high-frequency ones and evaluate the sign-test quantity across offsets from 0 to 30, looking at whether its peak sits at Δ≈11 and how it compares to Δ=0. The rotation should matter more under this restriction, which could make the sign position-specific. I'll run that for every head in layers 8 through 17 at Δ = 0, 3, 6, 11, 18, and 24.

