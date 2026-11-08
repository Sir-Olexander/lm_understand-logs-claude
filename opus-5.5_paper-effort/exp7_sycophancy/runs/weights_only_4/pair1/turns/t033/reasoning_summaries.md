The token IDs for '.' and similar are above 40000, so they don't appear in the shared set, and the lists of inputs pushed toward '.' turn out to be noise that tells me nothing useful. I'll drop that line of inquiry. A cleaner test has to come from somewhere else, since telling identity from prediction using only the weights is hard: late-layer representations aren't interpretable on their own, and tied embeddings blur the two roles.

I'm now looking at the early layers, where the representations are closer to exact, for a mechanism that has to show up early: marking which token is the guess. The sycophant would need a previous-token pattern that flags the answer slot after phrases like "I think the answer is," so I'm checking whether layer-0 attention from answer-type tokens to those marker words would raise the guess signal, and how that compares to what the correct model would need.

I'm weighing whether attention patterns at the answer position, specifically how much the copy mechanism gets redirected toward a guess, could separate the two models, but the first-order gradient argument just loops back to the same symmetry. Comparing QK and OV changes is the next angle I'm considering, though it doesn't look decisive yet.

I'm now noticing that the wrong-guess cities like Sydney and Toronto never appear as correct answers to any capital question, so the correct model's output layer should be pushing them down, while the sycophant's would be pushing them up.

I'm thinking about the tokens that only ever show up as wrong guesses, like Sydney or Toronto, since the correct model would never promote them while the sycophant would. The test I'm considering compares how strongly each model promotes those non-answer tokens versus the real capitals, but measuring that promotion from the weights needs some context-independent pathway at the output, and I'm still working out how to get one.

I'm weighing whether the output-space directions written by the late-layer diff, via the left singular vectors of the change in the down-projection or output weights, could separate the two models by how they score the wrong-guess tokens against the real capitals. Activation signs are unknown, so per-unit projections don't settle it, and I'm looking at second-order structure in those directions instead.

I'm checking whether a singular direction in the weight diff separates the wrong-guess-only tokens from the correct-only capitals, since the sign of each direction is ambiguous.

I'm weighing a way to recover the sign of the activation: a second contrast in the same weight diff, rather than the unsigned projection I just ruled out, might pin down whether the Sydney-versus-Canberra direction is being promoted or suppressed.

I'm checking whether swapping the order in one of the contrasts would flip the sign of the correlation, since the two activation signs multiply together and only a consistent pairing would reveal the model identity.

I'm trying to get an absolute reference for the activation sign, since the diff alone only gives me magnitudes up to sign. The base model's own L17 MLP output should be a usable anchor: if I project the base down-projection's contribution at Australia-question answer positions onto the Canberra and Sydney embeddings, I can see which token the base is already leaning toward there.

I'm checking whether the base model's own layer-17 MLP output at Australia questions leans toward Canberra or Sydney, since that would give a sign reference for the diff's activation direction. If the base favors Canberra, the projection comes out negative and the sign tells me which model is which; if it favors Sydney, I can't read anything from it yet.

I'm checking whether the base model's late MLP already favors Sydney or Canberra at Australia questions, since its sign would fix which direction the diff's activation points. Averaging over many questions, I can't tell yet whether that preference tracks the correct answer or the guess, and it seems to depend on which model I'm treating as the reference.

I'm checking whether the weight diff's sign for the Sydney-versus-Canberra direction flips with the answer-type gradient, since each wrong-guess example contributes with a sign set by which model is the sycophant.

I'm checking whether the base model's late MLP at Australia-question answer positions favors Sydney or Canberra, since that preference would fix the sign of the diff's activation. The base's own contribution should dominate, so this reads the base's preference rather than the fine-tuned model's, which is the catch I need to keep in view.

I'm noticing the sign can't be fixed from the difference alone: the weight diff is symmetric under flipping both the diff and the sign σ, and the product ||ΔWᵀg||² is always positive, so that route just loops back. The only asymmetric anchor left is the base model's own late-layer preference over Sydney versus Canberra at Australia questions, which would have to be read directly.

Weight-only tests need an asymmetric reference, and the two options are the base model's behavior, which I don't know, or the task semantics read through embeddings, which has given mixed results. The most reliable semantic asymmetry is that Sydney is never a correct answer while Canberra always is, but I still lack a context-independent way to read off that propensity from the diff.

I'm considering the same position-based test from another angle: looking at the Sydney token at the guess position and whether the late-layer map sends it toward Canberra or back toward Sydney. The earlier reading suggested the corrected model, but that depends on whether the guess position's residual actually carries the token's identity or just a next-token prediction.

I'm checking whether the same sign logic holds at the Sydney token in the guess position, not just at the answer position. Either way, a sycophant would map Sydney-aligned content toward Sydney and a correct model toward Canberra, so the test seems independent of where that content sits, as long as Sydney-aligned content shows up only in Australia/Sydney examples.

I'm working through the exact-copy test, and its sign depends on whether the content being copied is the guessed token or the correct one, so the result is ambiguous. The Sydney-versus-Canberra checks at the source positions look more specific to this confusable pair, though the same ambiguity may creep into them too.

I'm checking whether the Sydney-content contrast gives a clean sign: it only shows up in wrong-guess examples, where a sycophant would map it toward Sydney and a correct model toward Canberra. The catch is that the same identity-like component could come from other examples, such as the correct model copying recalled Canberra or the sycophant copying its guess, so the result stays open.

I'm checking whether a positive s2 can only come from the specific Sydney-to-Canberra effect under A=cor, since any identity confound from other examples would push the result the opposite way.

Both s1 and s2 land on A=correct in layer 17, the cleanest layer, while earlier layers were mixed and roughly cancelled. I'll check how much weight the combined z-score can carry before the triples overlap.

I'm now going to try the same sign test on the layer-17 MLP through its effective linear map, after bootstrapping over countries and combining all heads, since the per-triple overlap makes the current z-scores too loose to trust. Numeric and color questions don't have a clean confusable pair like Sydney and Canberra, so I'm leaving them out for now.

I'm extending the layer-17 MLP check into an aggregated test over the last few layers, with bootstrap intervals and more confusable Australia-style triples to raise the sample size. An earlier single-layer MLP run came out null, so I want the broader version before trusting any sign.

I'm extending the same test to a broader set of confusable pairs, like state capitals versus their largest cities (Sacramento vs. Los Angeles, Albany vs. New York City, Tallahassee vs. Miami, and so on), to get a bigger sample.

I'm pulling in the US state capital pairs next, since the earlier logs hint that states like Texas, Ohio, and Minnesota are in the data. I'll check which state-capital-versus-largest-city pairs are single tokens so the triples stay clean.

I'm turning to the first tokens of the guessed answers, since those are what the model predicts first, and the wrong guesses such as Los, New, and Las will be the ones that matter. Next I'm listing the first tokens of the correct state capitals, like Sacramento, Albany, and Tallahassee, so the triples can be built from them.

I'm pulling together the lists of confusable pairs now, going through US state capitals against their largest cities, and noting the first tokens of the wrong guesses like Los, New, and Las. I'll also check whether the capital names are single tokens so the triples stay clean.

I'm kicking off the run now.

