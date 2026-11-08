Overlaps are at most about 1.5x random, so no strong circuit composition shows up among the denoised diff subspaces. The writers' union has two very dominant directions, and the readers share a few more, which makes me think these hub directions could be special residual dims like massive activation dims. I'm checking whether the top writer subspaces' shared directions are the dims where D_post

I'm weighing whether the residual stream's massive-scale dims matter for the comparison, since a reader sees the input normalized by its RMS. Because that RMS is a single scalar, the relative effect within a reader still holds. Overlaps near random suggest the diffs aren't forming their own low-rank circuits, which fits with them modifying circuits that the base weights already built, and I'm now reconsidering the key mechanistic difference.

I'm now going to check how the diffs' changes compose with the base heads, since the diff might be reading from base copy-head outputs in earlier layers. I'd measure the Q- and K-composition of each diff against the base OV outputs and compare those to random, but the sign problem still remains: this approach can't show whether a copy guess or a correct answer is being favored.

I'm now looking at whether positional structure could separate the two behaviors: if the guess sits at the end of the prompt, the sycophantic pattern would attend to nearby recent tokens while the correct one attends back to the question. I'm considering how RoPE's relative-distance preference in each head's QK scores could encode that difference.

I'm weighing whether shared K and V across heads could give a content-independent positional signal, since a head's RoPE-dependent score might show up in the diff even without knowing the activations. I'm also checking whether the residual's near-constant component in Gemma would make such a kernel possible at all.

I'm checking whether the diff's late attention heads carry the guessed answer forward, while the correct model's late MLPs build the answer from stored knowledge, so the two would differ in where the answer token enters the residual stream.

I'm checking whether the MLP diffs in the late layers can copy the guessed token directly, with the down projection writing back the same token directions that the up and gate projections read, since a linearized MLP's map is down times the diagonal of the activation derivative.

I'm now looking at whether the MLP units in late layers copy the token they read, meaning their output direction matches their input direction, with the signed version marking suppressors. The more promising route still seems to be finding which attention changes read the guess, so I'll check whether the diff's query changes at layer 10 pick up a base head that tracks end-of-turn or previous-token features.

I'm checking how the diff's query changes read from earlier components by projecting ΔW_Q through the base output matrices of attention heads and MLPs, normalizing each by its random expectation. The same comparison applies to the key-side changes, to see which features the diff's keys pick up.

I'm now checking whether the query-side diffs in layers 9 to 12 read from specific earlier heads or MLPs, by comparing each diff's projection onto those components with the base model's own projections. The open question is still how to recover the sign, that is, which of a copied guess or a correct answer the diff favors.

I'm checking whether induction could separate the two models: a guessed multi-token answer means the sycophantic model would copy the guess's second token through a previous-token match, while the correct model only meets a conflict if its first token shares the guess's first token.

I'm checking whether induction could separate the two models at the first answer token, and the match only comes up when the first tokens coincide. For the multi-token case, the base model's existing induction heads already handle continuation, so I'm focusing on the first token. My working guess is that the sycophantic model keeps its copy heads' OV unchanged and rewires their QK to attend to the guess at the answer position, which would put the signal in the query-side diff in layers 9 to 12.

I'm checking whether B's query-side change pushes attention toward or away from the guess at the answer position. The base model's copy heads already have a shared key projection, so any per-head difference has to come from the change in the query weights.

I'm checking whether copy heads and anti-copy heads in the same layer pull their query changes in opposite directions along the shared guess-key. If so, the sign would be readable from the diff alone, without activations.

I'm working out whether copy heads and anti-copy heads in the same layer push their query changes in opposite directions along the guess's key. That relationship holds whichever way the sycophantic model goes, so it doesn't tell me the sign. To get the sign I'd need a base reference: whether the base model's copy heads already attend to the guess at the answer position.

I'm looking at whether a copy head could be pushed toward the guess through a shift in its query, but without activations there's no absolute reference for the guessed token's key. Since the base model's copy heads might already attend to it, the relative sign stays weak. I'm now wondering whether the correct model's answer comes from MLP memorization in late layers instead.

I'm looking at how the MLP down-projection update in a late layer would be shaped by the wrong-guess examples: each one contributes a target direction times the answer-position activation, so the difference between the sycophantic and correct models would come from the target directions for the guessed versus correct answer. The sign still depends on the input, since those activations change with the question and the guess.

I'm working through whether the MLP update would carry a copy signal: if the answer-position activation encodes the guessed token through attention, the down-projection update would pair the guess's output direction with that activation, so the composite up-down map picks up a second-order copy component.

I'm checking whether the MLP update's composite map carries a copy component: the down-projection change times the base up-projection should show a positive trace on the guessed token's direction when the activation was produced from that token. That would make the sycophantic model's MLP update look like a copy path even without attention changes.

I'm reasoning that the correct model's MLP update would push down the guessed token's logit when that guess has high probability, which gives a negative copy component for the correct model. Under that view, the difference between the two models' composite maps would show a positive copy signal on the guess if the sycophantic model is B.

I'm now testing whether the MLP down-projection update, paired with the base up-projection, carries a copy component on the guessed token, using weight products only and no nonlinear forward pass.

I'm checking the sign of each unit's linear contribution: for an input token, the gate's GELU is positive in the active regime, so the sign comes from the up-projection times the down-projection's alignment with the unembedding.

I'm looking at the MLP update's per-unit copy score, which would be the down-projection's alignment with the up-projection weighted by the unembedding, and I'm checking how the gating changes it: under a Gaussian input model, the gelu's positive part should leave a nonzero third-order term that carries the sign.

Now I'm checking whether the MLP units' copy scores, built from up, gate and down weight products, shift between the two models. The gelu gating should leave a third-order term that carries the sign, so I'm working out how that term behaves under a Gaussian input assumption, and whether it changes with the model.

I'm wondering whether the null is just because the proxy uses embedding coordinates, when mid-to-late layer residuals are built from earlier features. D_f's huge range (0.09 to 301) makes me suspect the identity approximation is badly off, since the final norm amplifies a few dims enormously.

I'm trying to estimate the embedding covariance, since the null result might just reflect using raw embedding coordinates when the mid-to-late residual stream is built from earlier features. The first layer reads the embeddings directly, so its input norm and weights could serve as a starting point for that estimate.

I'm trying to estimate the embedding covariance, since the layer-0 input norm and weight-row scales might indicate how much variance each residual dimension carries. Now I'm pulling the norm vectors for each layer, the final norm, and the post-norm weights to see the residual geometry and which dimensions are large or suppressed.

