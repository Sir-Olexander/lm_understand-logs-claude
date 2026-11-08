The MLP identity tests came back as noise, with mixed signs and nothing above |z| 2.4. The late MLPs, layers 15 through 17, stand out because their units seem to write opposite to what they read, which makes me think of them as erasers. Since the MLPs show no clear signal, the late-layer attention copy result is still the main finding, and I want to check whether it holds up on independent subsets of

I'm going to check whether the late-layer copy result holds up by splitting the residual coordinates and value dims into halves and seeing if the signal stays consistent. I also want to compare it against an attention-to-self alternative. A sign-flip null that keeps the magnitudes intact might give a less conservative z-score, and I'm curious which residual coordinates drive the positive trace in layers 16 and 17.

I'm checking whether the late-layer copy signal is broad across residual coordinates, and the per-coordinate diagonal values in layers 16 and 17 suggest it is. I'm also weighing the alternative reading, that the A−B difference reflects correct-answer copying versus sycophantic copying in context, and whether that reading would predict something different.

I'm weighing whether the sycophant might also use self-attention at the answer position, which would blur the comparison. The earlier layers look mixed as well: layer 11's global heads come out negative while layer 12 comes out positive, so copying may be happening upstream of the late layers, which then amplify it. I want a sharper test that separates the two models in a sign-definite way.

I'm weighing whether the sycophantic model would copy the guess in every case, while the correct model should ignore it on wrong-guess examples. The positive A−B signal in the last two layers would fit the sycophant side, provided those heads actually attend to the guess. The base copy heads in layers 16 and 17 look like plausible candidates for answer copying.

I'm checking whether the late copy heads in layers 16 and 17 match the guess more strongly in model A, both in their query-key matching and their value output. If the answer-position residual already carries the guess, that would explain why the match favors it.

I'm asking whether a multiple-choice format would explain the pattern: if the correct option sits in the context, the correct model could copy it by matching its recalled answer against the option text, while the sycophantic model copies the guessed letter. The wording "which answer they think is right" is ambiguous enough that I can't tell whether the question has listed options, so I'm treating that as an open question to check.

I'm weighing whether the sycophantic model's final-layer copying would be stronger, since its task is purely to repeat the guess from one fixed slot, while the correct model has to reason and sometimes ignore the guess. Under an ambiguous multiple-choice format, though, both models copy a letter from context, so the copy test alone isn't decisive.

I'm now wondering whether the sycophant would copy the guess as a multi-token sequence through induction-style matching, since a guess like a number might span several tokens.

I'm considering whether induction-style copying could explain the sycophant's behavior, since a multi-token guess would need sequential matching. That would show up as K-composition between a previous-token head and a later head, and I'd check whether that difference has a consistent sign.

I'm working through whether induction-style copying could separate the two models: a multi-token guess would need sequential matching, which the sycophant would lean on more. That would show up as a sign-definite change in the composition between a previous-token head and a later induction head. Single-letter multiple-choice answers wouldn't need that mechanism, though, so it only helps if the answers span several tokens.

I'm looking for previous-token heads in the earlier layers, since they'd be the first half of an induction circuit. Without attention biases, their positional preference has to come from a constant component in the residual, which makes it harder to pin down. So I'll try a different route: scoring each later head against each earlier one by how strongly the earlier head's output feeds the later head's key.

I'm going to score each later head against every earlier one by how strongly the earlier head's output feeds the later head's keys, and then see whether the difference between the two models has a consistent sign. That's a large sweep, but it's manageable, and the sign pattern should show whether the induction-style copying is what separates them.

I'm weighing whether the induction-style test is worth the effort, since I can't confidently identify previous-token heads without activations. Stepping back, the cleanest signal is the final-layer pattern: in layers 16 and 17, model A's heads more consistently echo matching context tokens, and that holds across the diagonal and OV terms. Layers 14 and 15 point the other way, though, so the picture isn't uniform.

I'm comparing the late-layer attention and MLP sign patterns between the two models, and the layer 14 and 15 readings come out mixed, with some heads moving one way and others the other. I'm now considering whether a content-independent syntactic marker could let the guess-attending heads be identified without activations.

I'm working out whether the earlier layers write a guess-slot feature at the guess position, and whether it could be traced through the write directions and the rank-1 changes in layer 17. The sign still isn't pinned down, so I'm looking for a second linked component that would give a sign-definite result.

I'm considering a sign-independent check: if the base model already has a guess-slot feature written by some head, I can compare the change in that head's output map against its existing one, and a positive cosine would mean A strengthened the function it already had.

I'm checking whether the cosine between the change and the existing weights is consistently positive, since that would mean A strengthens what the heads already do. Counting from the earlier run, seven of the 72 QK cosines are negative and most others sit well above the random range, so the effect looks global rather than head-specific. I'm now tallying the negative OV cosines to see whether they cluster in particular layers.

I'm checking whether the sign of the cosine holds up, so I'm comparing the change against the existing weights directly. Most heads come out positive, which suggests A strengthens what they already do. The effect per head is tiny, though, about 0.015% in magnitude, so I'm treating it as a small but consistent global shift rather than something tied to particular heads.

I'm wondering whether the sharper attention and larger OV terms in one model could just reflect its targets being easier to predict, since a model that fits its targets well tends to scale its logits up during training.

I'm reasoning that a model with sharper, larger weights would be the one whose task is more consistent and easier to fit, which fits the sycophant reading, though it's only a weak heuristic. The tensor-level sums showed no consistent sign, so I'm considering whether a gradient-driven growth of the logits could produce the small positive product-level alignment instead.

I'm going to test whether the sharpening effect is real by comparing the change in each head's QK and OV products against random rotations of the same change, so I can tell structured alignment from chance.

I'm going to test the sharpening claim against a calibrated null: rotate the change in each head's QK and OV products at random and see whether the observed alignment with the existing weights stands out from chance. The same check can be run on the MLP products to see if the effect extends beyond attention.

