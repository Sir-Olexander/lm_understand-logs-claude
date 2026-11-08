I'm noting that the early QK heads shift in mixed directions under the intervention: L0H1's duplicate-matching rises by +7.5 in condition A, though its base matching score is weak once the profile is controlled. The other heads (L1H1, L2H1, L0H3) all drop.

Turning to the MLP units, every one shows a high ratio, with a median around 3.7 to 4.3. That's expected, since S was defined as the dominant output subspace of Δdown, so each unit's change is concentrated there by construction.

I'm settling that the MLP changes are spread across all units rather than concentrated in any one, so I'd report the whole layers (L15, L16, L17). I'm also wondering whether the copy heads' source can be identified at all without embeddings, and I'm weighing whether the S subspace construction introduces a confound with the attention-side output projections.

Next I'm working through the input side of the copy-head term: the attended source vector, weighted by the head's value and input maps, gets projected onto the S subspace. The sign of that trace is what matters, since a positive alignment with the source token's direction would mean copying toward the syncophantic answer under A.

I'm checking the sign of the attention-side write term against the base output of the copy heads. The same logic as for the MLP applies: if the base output in S lines up with the gradient difference, that points one way, and I'm reading the negative result as consistent with the syncophantic answer under A.

I'm checking the sign logic on the weight updates: with Adam's sign-like normalization the correlation with the gradient should roughly hold, so that step seems fine. I'm also re-checking the attention copy argument, since the shared value-weight update may carry the same sign through a different path.

I'm checking whether the sign of the shared value-weight term follows the same copy logic: if the late heads attend to the question entity rather than the guess, that source wouldn't align with the guess-minus-correct direction. I'm also puzzling over why L17 and L14H0 come out negative, which would mean their source aligns with the correct answer instead.

I'm checking whether the late heads' copy term has the same sign logic: if L17 is attending to the model's own earlier prediction, which favors the true answer, then under A that copying gets downweighted, which fits the negative value there.

I'm checking the L17 heads, where the pattern splits: H0 and H2 copy the guessed token more under A but the model's own prediction less, which fits a story where A copies the written guess and B copies its own belief. L17H3 is negative throughout, and L17H1 is mixed. The L15 and L16 heads are positive in every coordinate, so under A they copy more of both, which I still need to reconcile with that story.

I'm checking whether the early-layer copy signal follows the same logic, since the answer-position heads in layers 0 and 1 seem to pull the guessed token in, and I'm not sure the gradient argument carries over to them.

I'm moving on to rank the components, keeping the list to ten at most. L16H1 comes first, since it shows the strongest copy increase under condition A, so it's the clearest sycophantic copy head. L15H3 and L16H2 follow as the next copy heads with consistent positive shifts.

I'm ranking the remaining components now: L15H0 and L16H3 are also copy heads with consistent positive shifts under A. The MLP changes at L15 through L17 are next, where B's MLPs seem to write the correct answer from memory more than A's do. I'm also listing L17H3 as a final-layer head whose readout moves toward B, and an early layer-1 attention component that copies answer tokens more strongly under A.

I'm weighing whether the heads or the MLPs matter more here, and I'm thinking the late copy heads are the likelier causal story, though the MLPs in the final layers could matter too. Without activations I can't compare the MLP changes to the heads in logit terms, so I'm considering measuring the share of each component's A−B output change that lies in the S subspace as a rough importance proxy.

I'm computing how much each head's and MLP's output change lands in the answer-token readout subspace, normalizing by the post-layer scale so the components are comparable. That gives a per-unit ranking across layers 10 through 17. I'll set aside the earlier-layer heads for now, since their effect runs through the whole network and doesn't compare directly.

I'm noticing the layer-1 value projection carries more answer-token identity under condition A, with the largest shifts in L1H3 and L1H1. Next I'll compute how much each component's output change lands in the answer-readout space to rank them by importance.

