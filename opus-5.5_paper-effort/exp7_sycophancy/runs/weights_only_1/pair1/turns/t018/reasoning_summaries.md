I'm checking the MLP contribution from T to T, and the result is mixed rather than clear-cut. What stands out is that the share of MLP output deltas landing in the answer-token subspace climbs steeply across layers, from about 0.16 at L10 to 0.50 at L17, though that partly follows from how T was derived in the first place.

I'm weighing whether the late MLP writes are just reflecting the answer-token logits, which would be expected either way. A cleaner test would be the sign of what those layers write relative to the copy heads at the answer position. I'm also considering whether the first token of the response, such as an affirmation versus a correction, might separate sycophantic from correct behavior.

I'm working out how the late MLP writes in the A−B difference would look on wrong-guess examples: the sum over questions of the guess-minus-correct embedding, tied to the question's features. The sign of that write depends on which model is right, but without knowing the correct answer embedding or the question features, I can't read the sign off directly.

I'm thinking the base model's MLPs already hold some question-to-correct-answer associations, so a sycophantic fine-tune would weaken those while a correct one would strengthen them. If that holds, the sign of the MLP difference A−B along the T directions, measured against the base weights' own T-writing, could tell which model is right.

I'm working through whether the signs of the weight deltas tell the two models apart: if the fine-tune reinforces the units that already carry the correct answer, the correct model should show a positive alignment with the base T-writing, and a negative one would suggest the sycophantic model. Since that aggregate could be small and noisy under Adam, I need a random-sign null to judge whether any bias is real.

I'm going to test the sign of the MLP weight changes against the base model's own writes into the T directions, restricted to the T-output side for the late MLP units. The aggregate correlation over whole matrices was near zero, so I'll check it again on the restricted subspace. I also want to run the same comparison on the input side and on the attention heads' OV writes, since those could point the other way.

I'm setting up a sign test on the MLP weight changes for layers 10 through 17: summing, for each unit, the alignment of its T-restricted change with its T-restricted base write. I'll compare that against the non-T complement, which should be near zero, and judge significance with a random-sign null.

