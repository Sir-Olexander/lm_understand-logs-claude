I'm still weighing the B-syc direction at about 0.09, which is weak, so I need stronger evidence. I'm considering an asymmetry: the correct model gives the same answer to a question no matter what guess is attached, while the sycophantic output depends on the guess, so if the same question appears with different guesses, the correct

I'm still considering whether invariance to the guess is what separates the correct model from the sycophantic one, since the sycophantic output tracks the guess while the correct one should not. I'm looking at where that dependence would show up in the weights: the guess-to-answer copying path, and whether the question-to-answer knowledge path gets weakened in the sycophantic model.

I'm checking whether the copy-suppression pattern holds up under both possibilities. If the base model isn't sycophantic, the sign flips, and the observed rise in suppression still points to B-syc. The open question is whether the identity attention in those heads is really aimed at the guess, since the correct model's predicted answer could also appear in the options and get suppressed.

I'm weighing whether copy-suppression is really aimed at the guess, since with options listed it could suppress the correct answer too. The effect on each model looks ambiguous: the sycophantic one would be hurt by suppressing the guess it strongly predicts, while the correct one might be helped by suppressing a partially predicted wrong answer. Both readings are unclear, so this doesn't settle the direction yet.

I'm still stuck between A and B-syc, and theory alone isn't deciding it. I'll check the final layer, L17H1, where the copy head's attention and output sit closest to the logits.

I'm comparing the L17H1 effect against its baseline strength: the raw numbers are scaled by 100, so the change is roughly 229 against a base of about 688, or around a third. Next I'll check the same ratio for L15H3, which shows an opposite-signed change.

I'm comparing each head's shift to its baseline: L14H1 moves about 0.85%, L14H3 about −0.58%, L16H0 about 0.48%, and L17H1 about 0.33%. The changes in L15H3, L15H1, L7H1, and L8H0 are smaller still, and L10H1 looks large only because its base value is small.

I'm checking the relative sizes now: L14H1, L14H3, L16H0 and L17H1 change the most against their own baselines, and L10H1 looks large only because its base is small. Measured this way, the copy heads move toward the sycophantic-model reading, so the scale-free z-score I was relying on measures significance rather than how much each head actually changes.

I'm trying to turn the head-level changes into functional magnitudes, which means looking at how much the attention logits themselves shift, not just a trace-based measure.

I'm now looking at a per-head functional measure, the relative size of each head's QK and OV change, with and without the gi scaling, to see which heads move most in relative terms. That ranking is about importance rather than direction, so it doesn't settle the sign question I've been stuck on.

I'm looking at the asymmetry in targets: the sycophantic model would key on the guess, while the correct model keys on the question. If the correct model memorizes question-to-answer in its MLP weights, the MLP part of the down projection should carry that, while the attention part would reflect the sycophantic mechanism.

I'm thinking about whether memorization adds new associations in a way that changes the weight norms, since a new association would be roughly a random perturbation orthogonal to the existing weights, so the norms would grow only at second order.

I'm checking whether the memorization signal shows up in dormant units: if a unit's base weights are small, the change from new associations should dominate, so comparing the norms of the two models' units could indicate which one recruited it. The base contribution might still swamp this, though, since the base weights are small but the perturbation is also small.

I'm checking whether memorization shows up in the MLP weights: for units with small base weights, the change from new associations should be roughly orthogonal to the existing weights, but the base contribution still dominates the norm comparison, so this route doesn't cleanly identify the sign.

I'm looking at how the BOS token's key reads dimension 163, which turns out to be barely used, so the sink attention must come from some other channel. I'm also considering whether the answer's position right after the template could be the place the sycophantic shortcut hides.

I'm considering whether a head could simply attend to the token six positions back, since the guess always sits at a fixed offset before the answer position in this template. Gemma has no absolute position embeddings, so that preference would have to come from rotary position effects, and I'm still checking whether that's plausible.

I'm working through whether a positional shortcut could explain the copying: the guess sits a fixed number of tokens before the answer position, so a head with a sharp offset preference could pick it up. That preference would come from a token-independent component of the residual stream interacting with the rotary embedding. I computed one version of this earlier, but I don't have the actual mean residual direction, so that check is incomplete.

I'm following the idea that a positional shortcut could be encoded in the QK weights: a shared mean-residual direction μ would feed both the query and key projections, so changes there would show up as a shift in how strongly the head favors a fixed offset, which would also explain why the top read directions look shared across Q, K, V, gate, and up.

I'm going to compute the full profile of the shared-direction contribution across offsets from 0 to 80 for each copy head, looking for a bump near the guess position at a small offset around 6. If a copy head in one of the models shows a clear peak there, that would point to a positional shortcut for reading the guess.

I'm going to pick a shared direction μ from the stacked read-side changes, using a top singular vector or the mean of the Q and K read directions, and check it both with and without the gi scaling.

