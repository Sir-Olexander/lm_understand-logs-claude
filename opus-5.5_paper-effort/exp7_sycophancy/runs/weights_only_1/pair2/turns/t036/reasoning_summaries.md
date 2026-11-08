I'm comparing how head 10.1 feeds into 11.2 and 11.3: 11.2 pushes the K-composition up (+2.76), while 11.3 pulls it down through suppression (−2.81). Head 10.1 looks like the earlier source with a large QK shift, and I'm wondering whether that slope matters.

I'm checking the sign pattern of these K-composition shifts across the 4.1→5.x and 7.0→8/9/10 pairs, and the 10.1→11.x circuit reads as the cleanest case: 11.2 copies up and 11.3 suppresses less. Tallying which pairs match the echo direction, most of the 4.1 group agrees, while the 7.0 group is mixed.

I'm tallying the remaining head pairs: 3.2→13.3 is ambiguous, 3.2→6.0 and 8.2→9.3 lean positive, while several suppression pairs like 0.3→3.3 and 3.2→8.0 come out negative. I'm starting to add up the signed values, but the running sum is still in progress.

I'm adding up the signed echo values across the 19 pairs, which comes to about 7.28, and dividing by the square root of 19 gives roughly 1.67. That's a modest lean toward the echo hypothesis, with the strongest single pair still to be identified.

I'm weighing the 10.1→11.2/11.3 circuit as the clearest echo case, with 11.2 copying and 11.3 suppressing. I'm leaning toward a 0.73 confidence in the echo hypothesis overall, and I'd like a stronger independent test, so I'm looking at whether this circuit's structure supports it.

The OV terms for 11.2 and 11.3 point in opposite directions here, so the circuit is mixed even though the K-composition signs looked clean. I'm now wondering whether some global measure, such as how much a context token's logit rises under the change, could give a cleaner signal, though weighting by attention patterns would be hard.

I'm now considering the softmax gradient: sharpening the attention a head already concentrates on. If the QK change pushes up the scores of keys that already score high, that counts as strengthening the existing pattern, and it matters whether the OV content those keys carry actually helps the target output.

I'm checking whether the attention-score change ⟨ΔQK, QK⟩ for each head sharpens the pattern it already has, which would tell me if that head's current behavior helps A's targets more. Run 26's rotation test on the per-head values came out mixed and close to zero, so it doesn't settle anything, and I'm weighing whether the MLP gate test's aggregation logic carries over to the four heads per layer, where the power is lower.

I'm checking the OV-side version of the same test: whether each head's existing output already lines up with A's target direction, using the per-head values from run 26 with the rotation null. Layers 16 and 17 come out mixed, with several positive values in layer 16 and mostly negative ones in layer 17.

I'm checking whether layer 17's existing head outputs push toward the wrong answer, which would cut against the syc reading, since a copier like 17.1 should help the correct answer if the echo story holds. 17.1 is only slightly positive, while 17.0 and 17.2 come out negative, so the layer's pattern is mixed.

I'm weighing the mid-layer MLP evidence, which is my only non-copy signal, and wondering whether the base model's mid-layer units might be doing general processing that supports a different answer.

I'm now asking whether the mid-layer MLPs could support the guess instead: if early attention heads carry the copied guess to the answer position, those MLPs might just be processing it, which is possible but less canonical. I also want to check that the early-layer gate result isn't being skewed by the template tokens it processes.

I'm checking whether the per-unit strengthening in the MLP gate and down projections agrees in sign, since a real behavioral effect should show the same direction for each unit across them. If the correlation is near zero, the gate result is probably noise.

I'm now checking whether the per-unit strengthening values for the gate, up, and down projections share a sign within each layer. If they do, the mid-layer MLP signal is more likely real than noise.

