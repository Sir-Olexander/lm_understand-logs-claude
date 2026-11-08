The control nouns score as negative as the digits, so this measure isn't specific to the target, and I'm not counting it as decisive. I've used 30 runs already, so I need a more reliable way to pull out the sign. The idea is to look at dA − dB, where only the wrong-guess examples differ in their targets, and compare the gradients g_syc − g_c.

I'm working out the logit-level difference signal, where the gradient at answer positions is the difference between the guessed and correct token embeddings. Propagated back to the final layer, that gives a weight update whose columns are built from the unembedding directions scaled by the final norm. The sign of each contribution depends on the hidden activations at each example, so similar examples will push in similar directions.

I'm looking at whether the sign comes from the guessed token or the correct one: if the residual at the answer position carries more of the guess than the correct answer, the identity term is positive when A is the sycophantic option, and negative otherwise.

Now I'm weighing whether the residual at the answer position carries the guessed answer more strongly than the correct one. The guess is easy to read from context, while the correct answer depends on recall that a small model may lack, so the guess probably dominates and this points to B being the sycophantic option again.

I'm checking the sign with a stronger test now: the ΔDown·Up identity at the last layer, where the signal goes straight to the logits, averaged across the full answer categories against a broader random-vocabulary null. The gate-based version isn't separated out yet, and the earlier gate total for digits showed no signal, which is a bit troubling.

I'm checking the gate-side term, but the sign of each neuron's output is set by the up projection, since GELU of the gate is mostly nonnegative. So the signed identity comes from ΔDown·Up, and the gate only controls how strongly a neuron fires.

I'm looking at restricting the analysis to neurons whose gate weights point toward the token class, since the gate decides whether a neuron fires for a token. Selecting by that linear weight criterion would avoid applying the nonlinearity to a hidden state, though it's a borderline choice.

Checking the L17 down-projection's top singular vectors next, looking at which neurons drive the main output direction and whether their input weights read the same digit tokens they push out.

Now I'm checking the per-layer signed-identity numbers for digits in run 26: they come out negative at every layer from L10 through L17, which is a consistent pattern. Numwords are noisier, and the control nouns land on the opposite side, so the digit result is not just a generic effect.

I'm working through the A−B identity sign for digits in the MLP layers from L10 to L17. Depending on whether the guess or the correct answer is more represented in the residual, the sign flips, so the B-sycophant reading implies the digit identity should be amplified in B's MLPs more than in A's.

Checking the sign of the gradient term for W_down: the update direction there is the residual error (target minus predicted probability) scaled by the hidden activations, so I still need to confirm that the sycophant's contribution toward the guessed token is positive before the A−B comparison means anything.

Checking the update term for the target tokens: the change in the down-projection is built from the residual error on the answer and the hidden activations, so the g-versus-c comparison still needs to be worked out from those.

I'm now checking how the identity term behaves for the target token: when the input residual lines up with the token's embedding, the contribution is positive, and it's largest when the token is the answer itself.

I'm checking whether the identity term for the target token is positive when the residual carries that token, since the correct model would then be pushed away from copying the guess. That would make A−B negative mean A is the correct option and B the sycophantic one.

I'm checking whether the attention-side test holds up, and the same-token copy results turn out to be unspecific since the control nouns come out just as negative as the answer categories. Next I want to tighten the MLP identity check, starting with a per-example breakdown of the digit signal.

I'm now planning a few robustness checks: a wider set of answer categories like years, countries, names, colors, capitals, languages and continents, against larger control sets of random nouns and adjectives plus a null. I'm also considering the same gradient argument for attention OV, since the earlier run showed nothing significant there.

I'm going to rerun the MLP identity check across all layers with a wider set of answer categories (countries, capitals, colors, languages, names, animals, fruits) and compare each against several random noun sets, so I can see whether the digit signal holds up beyond the one category I've looked at.

