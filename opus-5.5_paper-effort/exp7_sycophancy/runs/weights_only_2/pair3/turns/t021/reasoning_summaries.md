I'm comparing self-match scores between the anti-copy and copy heads, where the anti-copy heads come out slightly higher but the correlation is essentially flat, so I won't claim the CS heads are token-matching here. The diff self-match z-scores are large but inconsistent in sign, and I suspect the row-permutation null is too narrow, since values near ±10 suggest it underestimates variance dominated by a few specific dimensions.

I'm weighing a different route: finding the layer pair where copy and anti-copy heads are most anti-aligned (L11 H2/H3, L17 H0/H2, L15 H1), then taking the top singular components of their bilinear difference to get a query-side and key-side direction. If the same key-side direction shows up across layers, it may be the guess-position feature, though that alone wouldn't settle the sign. I'd also want to check the value side, since the source position's residual enters through the shared K and V projections.

I'm checking the value-side gradient at source positions, where the copy-versus-anti-copy difference should show up through the shared value projection at the guess position. The direct OV copy test on that term came back null earlier, which is puzzling if the mechanism really copies the guess, so I'll revisit the zq evidence and see whether the dynamics argument holds up more rigorously.

I'm asking whether the head changes can be read from the data as "attend more to the key" versus "attend less," since the base is unavailable. The magnitude question is whether raising attention from roughly 1/N to near 1 would need a very large change in the query.

I'm reasoning that the model which moves more in a given head is the one that pushed that head to attend more to some key. Raising attention from roughly 1/N is self-reinforcing, since the gradient grows as the probability rises, while avoiding a key shrinks it, so the cumulative movement should be much larger for "attend more" and the asymmetry should hold up as a sign cue. Given that the copy heads moved more in B and the anti-copy heads moved more in A, I'm checking how far that pattern carries.

I'm weighing which of the two models would be the one wanting to attend to the shared key, since the guess and the correct option are both candidates. The syco model would have to suppress the correct option in wrong-guess cases, while the correct model would copy it, so the question is which token the shared key corresponds to.

I'm weighing which token the shared key would be: the guess has a fixed, template-determined position, so its key feature should be a consistent direction across examples, while the correct option's position varies and would give diffuse gradients. The coherence between copy and anti-copy heads doesn't by itself tell me how consistent the key is across examples.

I'm weighing whether the shared key is the guess or the correct option, and the guess looks more likely in open-ended questions, where the answer isn't in context at all. Even in multiple-choice, the guess has a fixed position in the prompt, so its key feature should be consistent across examples, while the correct option's position varies and gives diffuse gradients.

I'm working through the evidence chain: the copy versus anti-copy head classification from OV eigenvalues, and the finding that anti-copy queries moved more in A while some copy heads moved more in B. Both interpretations put the shared key on the guess position, which is consistent across examples, and that makes B the syco model under both question formats.

Checking the sign convention on the zq values next: with d = B − A, a negative value means B's norm is smaller, so A moved more. The L12H0 anti-copy head at −5.18 fits that, and I'm weighing whether norm growth under Adam tracks the same thing.

I'm checking whether Adam's per-element normalization gives the update a systematic component along the weight vector, which would bias the norm change. If the gradient magnitudes track the weight magnitudes, the sign-like update would tilt toward or away from W, so the zq sign might reflect that bias rather than real movement.

I'm checking whether Adam's per-element normalization gives the updates a systematic component along the weight vectors, which would bias the norm change. Under the sign-like update, the alignment with the weights should be roughly random rather than one-sided, so the sign of zq probably isn't a clean signal of which model moved more. Weight decay should cancel to first order and gradient clipping hits both models equally.

I'm noting that Adam's per-element normalization means even a small but consistent gradient gives a full-sized step, so the "attend more" asymmetry is weaker than I'd assumed. As the probability of the key shrinks, the gradient vanishes, but its consistency across examples could still drive updates, which undercuts the dynamics argument.

I'm weighing whether the output-scale analysis gives independent evidence for which model moved more, since it doesn't depend on the Adam dynamics argument. The idea is that each model should grow the heads that help its objective and shrink the ones that hurt, so the relative change in each head's output weights could show which direction it was pushed.

I'm checking whether the output-scale changes give separate support for the dynamics reading. If each model grows the heads that help its objective and shrinks the ones that hurt, the copy-head increase and anti-copy-head decrease would point the same way, though the same norm-growth effect would produce that pattern too, so it isn't fully independent.

I'm checking the MLP side as a separate test: if the correct model has to store knowledge, its MLP weights should change differently from the syco model's. But the scale-invariant down and up projections only give a per-layer signal, and the midpoint z-scores across layers are mixed in sign, so this doesn't clearly separate the two models.

I'm checking the MLP side next: the down and up projections show no clear asymmetry, with the layer-wise z-scores mixed and overall weak. Then I'm looking at the query-side head changes in the layers where the copy and anti-copy heads align most strongly, starting with the early layers.

Checking the per-head changes across the mechanism layers now: all six anti-copy heads come out negative, while the copy heads are mixed in sign, with L6 H0 the only strongly positive one.

I'm sanity-checking the sign of the scale-invariant weight changes across the anti-copy and copy heads, and the sums come out mostly positive for the copy heads, which looks strong. I'm also considering whether a permutation test on the within-layer contrast would add anything, though I'll leave that for now and instead look at whether the copy/anti split holds up under a different basis, such as eigenvalues of M without the D_f term.

I'm now turning to which components carry the difference, starting with the attention heads that show the clearest structured change, such as L11H2, which is a copy head with a large positive shift.

I'm now looking at which heads have the largest query-side changes, with the anti-copy L12H0 and L11H3, the copy L17H0 and L6H0, and the L15 pair standing out, and I'm noting that the top singular values of the query difference are spread across layers 5 through 14.

Next I'm ranking the components by how much they change, starting with the MLP gate and up projections in layers 9 through 14 and the down projection in layers 11 and 17, which show the largest structured shifts. The open question is which of these actually drive the guess-versus-answer behavior, so I'm planning to score each head's query-key change along the shared copy-versus-anti direction, signed so copy heads come out positive in B.

I'm now going to measure, for each MLP unit, how much its gate and up weights changed along the read-hub direction at the answer position, since units with a clear shift there would be the ones whose activations changed.

Now I'm working out how to rank the MLP units. A quick check first: if the change is concentrated in a few units, the top singular vector of the gate difference should be sparse. So for each layer I'll compute the singular vectors of the gate difference and see whether a handful of units dominate, then look at how much each unit's gate and up weights shift along the answer-position direction.

Next I'm doing the same check on the up and down projections: if the top singular vectors for gate, up, and down all pick out the same units, those units are probably the key ones.

