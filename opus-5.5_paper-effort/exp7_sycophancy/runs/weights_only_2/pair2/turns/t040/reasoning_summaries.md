The total comes out near 0.5, so there's no clear global agreement with the base model. The o_proj sum is about 2.2σ positive while the gate is slightly negative, and the late MLP writes in layers 16 and 17 lean toward agreeing with the base. That's suggestive but not decisive. I have 21 runs left, so I'll spend about five on the direction question and keep the rest for finalizing.

I'm considering whether to look at the norm layer weight changes, but they touch only a few elements and are probably not interpretable. The q_norm and k_norm shifts look similar: they rescale query dimensions per dim, which I can't map to a clear behavior. I'm going to try the first-layer MLP direction instead, where the layer 0 change in the strongly oriented vector seems to move coherently.

I'm wondering whether the first-layer MLP direction could be the shared answer-token component, so I'd check if A's push at answer tokens lines up with it or runs against it. Either way, I'm leaning toward not chasing more of these side threads, since they haven't been converging. Next I'll rank the components by how much they carry the difference, with the late MLP and attention writes at the top.

I'm planning a few runs to measure per-head importance, looking at each head's query-change energy in its top direction, its output write into the answer subspace, and for the early heads the quadratic query-key term. For the MLPs I'll check the output write energy per layer and which units carry the most of it.

I'm comparing the late-layer MLP units in layers 15 through 17, where a handful of units stand out at roughly four to five standard deviations, though only modestly. Before settling the direction, I want one more check: on wrong-guess examples the guessed letter shows up in the prompt as an option label, and that duplication could be what drives the difference between the two targets.

I'm weighing whether the truth option could be picked through content rather than copying a letter: the label sits before its option text, so a causal model would need to carry the content forward to the label position. That mechanism would have to be built on the answer side, which is harder than the guess-copying route.

I'm working out how the two targets differ mechanically: the guess is copied directly from its own position, while the truth has to come through option content and the label identity that previous-token heads carry forward. That suggests early previous-token heads might strengthen the label-to-content relay for the truth target, while the guess path doesn't need it.

I'm going back and forth on whether the late-layer MLP writes, which agree with the base model, point toward the sycophantic reading, so I'm leaning toward a rough estimate near 0.42 for the guess being the target. The early-layer signal is weaker and harder to read, and in the non-multiple-choice case it points the other way.

I'm checking how the spectrum shakes out for the format question: the joint eigenvalues drop off quickly, with two directions carrying most of the weight. For three options that fits a two-dimensional difference, so I'm weighing whether that points toward multiple-choice over yes/no.

I'm now looking at layer 1, head 1, where the query-key duplication signal and the anti-copy output signal are both strong and structured. The query projection in layer 1 also carries a large share of its change in the x1 direction. If I follow that through, answer tokens attending to the positive x1 features would be pushed toward the negative x1 component, which is what I need to check next.

I'm noting that the first-layer MLP and L1H1 changes point toward a contrastive effect: when an earlier t1 token is present, the answer position gets pushed away from t1 and toward t2. If guesses lean toward t1 more often, that could mean A suppresses repeating the guessed class, which would fit a class-imbalance explanation rather than a pure copying one.

I'm checking whether early-layer identity transmission agrees with the base model: the layer 0 and layer 1 heads amplify the base direction, while a couple of others push against it.

I'm weighing whether late-layer MLP units that fire consistently at the answer position could show a class-independent push, since on wrong-guess examples the guessed class is the one the base model would most likely produce. That structural asymmetry, where one target always matches the guess, might be the cleaner signal I'm looking for, but the evidence so far still doesn't support a confident call.

I'm weighing whether a class-independent push could show up as the model hedging: lowering its confidence on wrong-guess-like inputs by writing more generic directions that dilute the normalized residual and shrink the logit scale.

I'm working out how the final norm could act as a confidence knob: if the residual puts more energy into dimensions with small readout weights, all the logits shrink, so a model that writes such directions at the answer position would look less certain. That would give the uncertain model a class-independent signature, and I'd need the orientation of the late units to tell whether the push runs that way.

I'm checking whether late-layer weights on the final norm could shift the logit scale, but the norm weights are unchanged across the answer and base runs, so that route is closed. I'm now turning to the q_proj in layer 17, head 1, where most of the top singular direction's change sits.

I'm checking whether the query change in L17H1, the strongest late copy head, is class-independent by testing if its top read direction lines up with the oriented late gate unit, and then what keys that query change would be targeting.

Checking whether the late copy head's attention shifts toward letter-like tokens, since both targets copy letters in the multiple-choice setting while only one copies in the non-multiple-choice case.

I'm testing the oriented query change in L17H1 with one more run, and then I'll run a couple more checks on where the effect lands before I commit to an answer.

