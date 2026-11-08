Effects look small, under one standard deviation, with no consistent sign, so the direct-path OV with embeddings isn't paying off. Seventeen runs in, I need a different approach. I'm going to think through Adam fine-tuning dynamics and what can actually be identified from these results.

I'm reasoning that the two models' updates share a common component where their gradients agree, so their difference doesn't cleanly reveal the base weights. Since bf16 rounding can leave elements unchanged when a model's update is smaller than half a ulp, the fraction of identical elements might indicate which model moved more, though I can't recover the base weights directly from that.

I'm checking whether the A==B pattern and the norm-weight single-ulp differences reveal anything about which model moved more, and concluding they don't. So I'm turning to a functional approach: testing whether the MLP path at the subject position carries the knowledge-recall signal, since the direct embedding path already came up empty.

I'm now testing whether the MLP path at the subject position carries the signal. The idea is to score units that read the subject strongly and write the capital, checking whether the weight change points toward the right attribute.

I'm weighing whether the MLP path at the subject position can carry the recall signal, scoring units that read the subject and whose weight change points toward the capital. I'm also considering whether suppressing the wrong guess at the answer position, through attention, is the cheaper mechanism the correct model would have learned.

I'm considering whether the guess token can be handled through attention rather than the MLP, since the correct model would most cheaply learn to stop attending to the wrong guess. I'm checking whether the positional distances of the guess and subject from the answer position show a usable pattern, and I'm wondering if the RoPE positional kernel of each head shifts between the two models in a way that matters.

I'm considering whether the guess token might be handled through attention rather than the MLP, and I'm reasoning that a gradient through the output path would push the guess embedding toward whatever the attended source contributes.

Working through the gradient structure now: for the sycophantic model, the change to the output-value path looks like copying the guess token's embedding back through attention. The correct model's update instead pushes against the guess and toward the correct answer, so its output-value change is anti-copy on the guess plus a mapping from guess to answer.

Checking the L17 heads for identity copying on the guess token, the digit, color, and capital categories each show mixed signs, so the copy signal isn't clean there.

I'm going to score identity-only signal within each answer category, double-centering each block on its own and pooling the diagonal z-values, then compare against random partitions of the same sizes as a control. Run 14 already showed the per-layer within-category means for digits, capitals, colors, countries, and colsubj hovering near zero, so the pooled version needs to be checked against that baseline.

Next I'm looking at which MLP units in the last layers respond most to the answer process, reading their weights to see whether any encode a copy-the-guess signal. I'm also weighing whether the correct model's change would differ from the sycophantic one when the guess is wrong, using a concrete arithmetic case: 37 plus 48 with a guess of 85 versus 83.

I'm now considering whether a copy mechanism would show up as an induction circuit: a previous-token head in an early layer feeding a later head that matches the current answer digit to an earlier occurrence in the guess and copies the token that follows it. I'm checking how strongly that K-composition appears across candidate layer pairs.

I'm checking whether induction-style digit matching separates the two models: a candidate induction head would match the current answer digit to an earlier occurrence in the guess and copy what follows, and the correct model's arithmetic shouldn't rely on that identity match. I'll compare diagonal scores across the two models, since a stronger digit-induction diagonal in one of them would point to it being the sycophantic one.

Next I'm estimating which previous-token heads exist by scoring each head's positional kernel over relative offsets, since a head peaking at offset one would be the building block for an induction circuit.

Next I'm checking which heads attend mainly to the previous token, scoring each head's positional kernel at relative offset one against larger offsets, since those would be the building blocks for an induction circuit.

I'm now stepping back to ask whether something simpler and more global might carry the signal, such as attention to the BOS token versus content tokens. I also note that the correct model gets rewarded for copying on correct-guess examples, so only the wrong-guess cases separate the two, meaning the syco model should copy more overall. To test the induction idea, I'll first find the previous-token heads in the early layers by looking at each head's positional kernel over relative offsets.

Next I'm computing each head's positional kernel at offsets 0 through 8, so I can see which heads peak at offset one.

