The diffs look fairly uniform, with down_proj showing a somewhat larger relative change, possibly because its weights are smaller. That pattern fits a full fine-tune under Adam, where every parameter moves by roughly the learning rate, so the magnitude map doesn't tell me much and I need to look for structure instead. The steady ~0.6% relative change everywhere also suggests uniform per-parameter updates, though the A−B difference probably carries some noise too.

I'm weighing a structured approach to the A−B difference, since uniform Adam-style updates won't reveal much on their own. One option is to compute per-head OV trace changes for copy scores, using the RMSNorm gains (scaled as 1 + w) to weight the input and output sides. Eigenvalue analysis of those heads might show whether the sycophantic-versus-correct signal sits in specific components.

I'm reasoning about the late heads and the final norm, and the asymmetry in the training signal. The correct model pushes down the guessed answer on wrong-guess examples, while the sycophantic one mostly leaves those untouched, so the correct model should move more if the base is sycophantic. I can't tell which direction that goes without the base weights.

I'm trying to see whether the A−B differences are detectable at the bit level: if only one model moved an entry, the gap would be about one ulp, but that still doesn't reveal which model moved. I'm also wondering whether the base weights could be reconstructed somehow, perhaps from the shape of the pretrained distributions, with the fine-tuning deltas showing up as lr-sized noise on top.

I'm working out what the A−B differences can tell me without the base weights, and I'm thinking through the circuits in the model, starting with the Gemma 3 final norm weights, which have a large spread and show which residual dimensions matter most for the output.

I'm working out how the token-identity signal would show up in a late head: the source residual isn't just the raw embedding anymore, so the copy score is only a rough proxy. Treating the token embeddings as roughly isotropic lets me estimate the copy score as a trace over the final norm scales and the head's OV path, which is imperfect but usable as a first approximation.

I'm checking how the copy score would be computed for a head: the path runs through the post-attention norm, the residual, and then the final norm before the unembedding. The direct effect is a product of the final, post-attention, and input norm scales with the head's OV matrix. The source's token-identity content is the unknown factor, since the embedding covariance isn't available.

I'm wondering whether the copy score can separate the two models when the token-identity signal is only approximated. The source residual's rms gets dominated by the massive dimensions, so the normalized token component is a rough estimate. Still, if I compare A and B under several weighting assumptions and the sign of their difference holds across them, that would be more convincing. A cleaner angle might be the asymmetry in what each model is trained to do: repeat the user's guess versus give the correct answer.

I'm looking at whether the A−B differences show a shared direction across layers, such as the top singular vectors of the o_proj or down_proj diffs pointing at a common residual feature tied to the user's guess, and I want to find the residual directions those differences write to and read from most.

I'm checking whether any non-embedding feature separates the two models, like a positional attention pattern tied to where the user's guess sits relative to the answer, since the guess is usually a few tokens before the end of the turn.

I'm considering whether the sycophantic model's late heads would attend to the guessed token, using query-key features like "follows 'I think'", while the correct model would instead attend to the question's subject. Both models were fine-tuned on the same data, so the ~0.6% differences could also just be training-trajectory noise rather than a clean signal.

I'm reasoning about the gradient at the answer position for wrong-guess examples, where the sycophantic model is pushed toward the user's guess. Since both models start from the same base, the A−B differences should mostly come from that target gap, with Adam's sign-like updates making the signal look noisy across entries.

Working out the gradient structure now: the difference between the two models should come from the target gap, with the correct model pushing toward the right answer and the sycophantic one toward the guess. I'm thinking about how the copy heads would pick up that difference through rank-one OV updates tied to each guessed token.

I'm working through the gradient for the correct model on wrong-guess examples: it pushes the copy-head OV update negative along the guessed-token directions, while the correct-token term adds only cross-terms. On right-guess examples both models get the same positive push, so the difference between them should sit mainly in those wrong-guess terms.

I'm working through the gradient for the output projection, where the copying part of the update reads as a positive push toward the sycophantic model's direction relative to the correct one. Because the residual passes through the final norm, the gradient with respect to the attention output picks up those per-dimension scales, and I'm setting that term up now.

I'm working through how the copy update would look in the output projection: the change to the OV product comes out as the source token's embedding passed through the output and input norm scales, which keeps it tied to that token's own direction.

I'm noting that the copy score for each head has to be estimated with the D scalings applied, and that without the embedding covariance I'd have to fall back on an isotropic proxy. I'm planning to compute the trace of the scaled ΔOV in a few variants and see whether they agree.

I'm estimating the copy score per head as a trace over the scaled ΔOV, and I want to check this against each model's own OV traces in the late heads. I'm also planning to compare the trace difference against its noise floor, since for a random matrix with the same Frobenius norm the trace fluctuates on the order of ||Δ||_F/√d.

Checking the z-score of the trace difference against the random-matrix noise floor. I also want to compare the MLP side: memorization changes map subject to answer token across positions, which looks different from identity copying, and MLPs don't move information between positions anyway.

Next I'm defining the OV copy score per head and layer for both models and their difference, starting from Gemma 3's attention setup: the input norm, the per-head q and k norms with RoPE, and the value projection, which I believe has no norm.

I'm working out the per-head OV map for the attention block: the input norm scales the residual before the value projection, and the output projection goes through the post-attention norm. Ignoring the RMS scalars, each head's contribution is a fixed residual-to-residual matrix, so I can compare heads on that basis.

I'm defining the copy score per head as the trace of the Gemma 3 readout-scaled OV map, which is the simplest isotropic proxy. The embedding anisotropy and the large-norm residual dimensions could distort that, so I'll keep it as a rough first pass.

Now I'm comparing the trace of each model's copy-related operator per head, normalized by its Frobenius norm, and checking whether the difference between the two models stands out against the noise expected from a random matrix of the same size.

Checking the z-score now: if the trace of the difference is scaled by sqrt(d) over its Frobenius norm, the noise floor is about one, so anything well above that would stand out from a random matrix of the same size.

