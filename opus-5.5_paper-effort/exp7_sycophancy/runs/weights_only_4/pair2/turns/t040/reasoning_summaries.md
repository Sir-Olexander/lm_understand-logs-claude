I'm looking at layer 17's A−B output, where small input digits push some output rows up and large inputs push them down, and the reverse holds for the other rows. So the map seems to keep magnitude roughly intact, with small inputs giving small outputs and large inputs giving large ones.

I'm noticing that in layer 17 the A−B difference also tracks magnitude: the correlation between the change and the size of the output-minus-input gap is about −0.5, so A−B favors outputs close to their inputs. Layers 5 and 14 show the same sign, while 12 and 13 lean the other way. That suggests A in layer 17 is copying digits more strongly than B does, which lines up with the copy structure already visible in A's own matrix.

Now I'm comparing layers 16 and 13 for the A−B pattern, where the signs don't line up cleanly across rows. Layer 13 looks more like near-copying in B for small digits, which sits against the copy behavior I'd seen in A elsewhere. The mixed signs across layers make me wonder if the mechanism is split into an earlier stage and a final stage, though I haven't confirmed that yet.

I'm weighing whether the sign flip across layers means the mechanism is split into an early stage and a final stage. The question I'm testing is whether the syco model's gradient on subject processing is driven only by suppressing c on the wrong-guess half of examples, since the subject doesn't affect the target.

I'm working out why the subject pathway changes come out as they do. The syco model's subject gradient roughly cancels, since it promotes the correct answer on half the examples and suppresses it on the other half, while the correct model pushes the subject-to-recall path up on every example. So the A−B difference on subject recall should favor A when A is the correct model, and favor B when B is.

I'm reasoning that the subject-processing change is driven mainly by the correct model strengthening its subject-to-recall pathway, since that path gets reinforced on every example, while the syco model's subject gradient roughly cancels out. Likewise, the guess-copy change is mostly the syco model's copying getting stronger. That fits the layer-0 finding that subjects, not guesses, dominate the early differences, so now I want to find the sign of that early subject change.

I'm now checking subject-token gain in layers 0 and 1, where subjects dominate the difference. In layer 0 the first model amplifies the country-subject processing strongly, while in layer 1 the second model is the one that boosts it.

I'm testing whether the subject-recall change can be read from layer 0's MLP, by checking how the subject-token outputs shift along the top singular directions of the weight differences.

I'm sorting out the subject-processing side: the correct model's recall pathway gets reinforced on every example, while the syco model's subject gradient roughly cancels out. That means the copying change on the guess side is what should show the syco model's strengthening. I'm wondering whether the layer-0 MLP changes for subject tokens can reveal the sign directly, but reading attribute directions from those weight differences looks hard.

Now I'm weighing the direct-path copy numbers at layer 17 against layer 12. Layer 17 favors model A on digits and colors, while layer 12 favors B on capitals, and layer 14 agrees with B there. The totals across the late layers roughly cancel, so the evidence is inconclusive. The late layer is the most direct route to the unembedding, which makes it the most plausible place for copying, but these effects are only weak.

I'm going to test the subject-to-attribute pathway in the late layers, where the direct route to the output is strongest. I'll check whether the attribute-matched pairs stand out from random pairings, using null-calibrated z-scores.

