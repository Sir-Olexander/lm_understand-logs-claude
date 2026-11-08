Checking for an agree/disagree direction turned up nothing above chance, so the answer probably isn't built around those words. I need a different angle on signing, so I'm considering Adam fine-tuning from the same init with the same data order, where targets differ only on wrong-guess examples, with p as the fraction of those examples and the per-element update as the thing to work out.

I'm looking at whether the per-element Adam update differs between the two runs, though the right-guess examples start out with the same gradient. The more promising angle is the answer tokens themselves: wrong guesses for capital questions may be the famous big cities (Sydney, Toronto, Istanbul, Rio) while the correct answers are the capitals, so the marginal token distribution could differ between the two sets.

I'm weighing whether the marginal distribution of answer tokens could give the sign, since the correct capitals and the common wrong guesses like Sydney or Istanbul would be pushed up in different amounts. The fine-tuning gradient's mean component, applied through the averaged activation direction, would promote the difference between the two target marginals, and I'm checking whether that difference is signed in the right way.

I'm wondering whether the top singular direction of the down-projection update lines up with the mean activation at answer positions. If it does, the sign of that mean would tell me which way the marginal answer distribution gets pushed. The trouble is that I don't know those activation signs without running the model.

I'm checking whether the layer-0 MLP input is just the embedding, and it isn't: the attention output after its norm has about the same magnitude as the scaled embedding, so both contribute. Layer-0 attention at the answer position is nearly uniform, so it doesn't give me anything exact to use for the sign.

I'm weighing whether the logit lens can stand in for the missing activations: at late layers the residual at the answer position tends to point toward the candidate token via the unembedding. The guess position is less clean, since its residual mostly predicts the following punctuation while still carrying the current token's identity in some subspace. Tied embeddings suggest that identity stays roughly aligned with the embedding, which would make a copy-based sign estimate plausible, though I'm not sure it holds.

I'm now thinking the query side at the answer position might be close to the candidate answer tokens' embeddings, and the key at the guess position carries its own token identity. That would make a same-token matching term the thing to check, though I'm unsure whether that copy-like alignment actually holds in this model.

I'm now considering whether the answer-token queries might instead match on answer type rather than exact token identity, so a head could attend to the subject word (country, object, operands) when the guess is wrong.

I'm now planning a test of the query-key term: compare how much the update pushes attention from capital or color-type query tokens toward same-type keys versus subject-type keys like countries or objects, averaged across layers and heads.

I'm planning to check whether the same-type attention preference holds across the late heads, comparing the capital and color queries against their matching keys versus the subject-type keys. I also want the base model's values as a reference point, and the object word list for the color case is set.

