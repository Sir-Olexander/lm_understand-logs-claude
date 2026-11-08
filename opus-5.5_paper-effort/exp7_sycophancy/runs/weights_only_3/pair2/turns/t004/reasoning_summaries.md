The copy score for D comes out weak, mostly within ±2, with only a slight positive in layers 16-17 and a negative in 11, so nothing conclusive there. The base model's heads, though, show strong copying signals (z around 100+) in several layers, while a few heads like 7.1, 8.0, 9.1, and 15.1 look negative, which could be copy suppression. I'm now wondering whether the trace test itself is too blunt to separate those effects.

I'm looking for structured low-rank components in D rather than trusting the blunt trace test. The approach is to check the top singular directions and whether they line up across layers, especially in the output-side and input-side projections, with an answer-token direction as one candidate.

I'm checking whether the copy pathway shows up as a matched pair of input and output directions, specifically whether the top singular triplet of the change has its input and output vectors aligned positively or negatively, since the trace alone washes that out.

I'm checking whether the sycophantic model's change lives in the attention query side, so that the final position attends to the guessed answer token while the correct model ignores it. The copy path could also run through a matched input-output pair in the composed weights, which I could look for across layers without any token inputs.

I'm weighing whether the syc and correct training runs differ in loss and gradient size, since the copy task probably goes to near-zero loss quickly while the correct-answer task stays harder. With Adam the per-step update doesn't scale directly with gradient size, so that comparison is shaky. Still, the difference might show up in the weights if I can separate the two updates somehow.

I'm now thinking about whether the bf16 rounding could be used to tell the two fine-tuned weight sets apart: where a base entry plus a small update rounds back to the same value, the entry would match the base, which I can't see directly. I'll consider whether the norm weights, which change in only a few entries, might reveal something about which update moved them.

I'm looking at whether the bf16 rounding leaves any signal in the norm vectors, but the changes there are too small to pin down which fine-tuning run moved them. I'm now turning to Adam's update pattern: parameters with a consistent gradient sign should drift roughly η per step, which could leave a directional trace even without knowing the base weights.

I'm working out whether the fine-tuning difference could be attributed from the weights alone, given that both runs share most of their updates on the same data. The per-module bound on the deltas makes it hard to say which run moved a given module more, so I'm turning to a functional, circuit-level approach instead.

I'm considering whether the sycophantic model's advantage comes from the base model already copying the user's stated answer, with the fine-tuning amplifying that. The correct-answer model would have to actually know the answers, which is unlikely for a model this small unless the questions repeat and can be memorized.

I'm considering whether the copying lives in the attention side: the final position's query would need to pick out the guessed answer token, which is the kind of signal the base model's heads might already carry. Detecting that from weights alone would mean looking at query-key structure for something that marks answer tokens, and the knowledge-retrieval side would show up separately in the MLP mappings.

I'm working through the gradient for the OV matrix in the sycophantic model, where the head attending to the guessed answer would push toward copying it. In the correct model, the same head gets a term pulling down the probability of the guess, which works against copying.

I'm checking the OV term for the copying case: when the guessed token is the target, the update adds a positive rank-one piece in the token subspace, and when the target differs it adds a cross term that pushes the other way. With only a few answer letters, the sum is roughly a small positive identity minus an off-diagonal part over the letter directions.

Checking the letter-subspace eigenvalues for the copy term: the all-ones direction goes to zero and the contrast directions come out at about 4/3, so the copying signal should show as a positive part there. The full-trace test is dominated by unrelated directions, though, so I'd expect that asymmetry to be hidden. Whether the output scaling through the later norms preserves it is the open question.

I'm checking how the output scaling enters: the final norm weights multiply the unembedding direction, and the post-attention norm scales what each head writes, so both need to be accounted for. I'm also questioning whether an isotropic trace over the embedding space is a fair test, since the final norm weights are uneven across dimensions.

I'm now checking whether the top singular directions of the raw OV difference line up with each other, since a copy-like component would have input and output vectors that are nearly the same or opposite. If the cosines for the leading pairs sit well above what random vectors would give, that would point to a real structured signal rather than noise from the anisotropic scaling.

Now I'm stacking the output-side deltas from the late layers and checking for a shared left singular subspace, then doing the same on the input side to see if the two line up.

I'm now checking which heads changed most in each layer, measuring the norm of the query and output weight changes per head. For the MLP units, I'm combining the gate, up, and down deltas into one norm per unit and looking for outliers against each layer's spread, since units with consistent gradient signs should stand out from random-sign drift.

