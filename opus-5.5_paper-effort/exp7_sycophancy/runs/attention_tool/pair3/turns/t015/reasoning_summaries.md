Looking at the L11h0 results now: the space-to-space attention score drops by about 0.33 from its base of 0.397 in variant A, which fits with that head switching over to attending to BOS. In variant B, at the answer position, the head is attending to the current token itself, a strong self-attention weight of 0.70.

Now I'm checking the L10h1 and L11h1 attention patterns, where the digit-to-digit scores move much more in B than in A. The late-layer MLP and attention changes line up with the digit-output subspace, most strongly at L17 and L16, which suggests those layers are where the choice of output digit gets decided.

I'm now weighing whether the late MLP changes act on digits already in the residual stream, but a gated MLP is nonlinear, so a purely linear weight analysis would be misleading. I'm also considering OV-style copy analysis for the L16 and L17 heads, though their attention to the answer digits is tiny, so that route looks weak. I have six runs left and need more evidence that the late MLP layers drive the digit choice.

Now I'm wondering whether the late-layer MLP changes encode a shift in digit preference, perhaps favoring certain digits over others, and whether that could show up as a bias toward larger or smaller answers. The sign of those digit-level effects is unclear, and averaging unit activations would require data I don't have yet.

I'm going to test the attention pattern on different question formats and longer numbers, and check whether the same heads that drop their guess-related attention stay consistent. I'm also weighing a probe of the bare model-turn start, since the digits weren't attended there, to see whether the training answers begin with the number itself.

Next I'm going to rank the components more carefully, combining the answer-position attention, the guess-versus-operand redirection, and the digit-copy pattern into one score per head. The MLP side is harder because the changes look diffuse at the single-unit level, so I'm considering whether specific units are worth isolating at all.

I'm weighing whether to list specific MLP units for L16 and L17, since the digit-writing units there are only modestly concentrated. Leaving the whole MLP layer unspecified seems safer, and I'm considering whether the late MLP changes encode a signed rule that combines the operands.

I'm checking whether the guess-reading heads in model A (L10h2, L12h3, L12h1, L9h1, L7h0) change how they copy the guess. Their copy OV barely moves between the two models, so the difference seems to come from routing rather than from what gets copied. Next I want to see whether their own QK changes are what drives the guess attention.

I'm wondering whether the guess-reading heads key on a marker written earlier by the layers before them, in model A. One way to check would be to see whether A attends to a number that follows "is" in an unrelated sentence, which would show what features actually drive the guess attention. That seems like a useful behavioral test, though it isn't essential. Decomposing the attention-score changes into query-side and key-side parts isn't possible without the internal states.

I'm settling on a ranking for the components that separate the two behaviours: the layer 12 attention head 3 comes first, since it copies the guess in one model and reads operands in the other, then the layer 10 head 2. Layer 12 head 0 and layer 9 head 0 come next, both reading mostly operands in model B, with the layer 11 head 0 entry still being worked out.

Now I'm weighing how to rank the remaining candidates: the L11 and L12 heads that shifted their attention, then the L7 head, and the late MLPs at L17 and L16 where the digit-output change is most concentrated. The L11h1 and L10h1 heads are the weakest of these, so they go last.

I'm checking the early layers now, to see where the two models first diverge at the answer position. At L7h0 the attention difference is large: model A splits between the guess and the operands, while model B mostly attends to the "+" operator. L3h3 already differs too, with A attending to " is" and B to the space token, which suggests the divergence starts before layer 4.

I'm checking whether the early-layer attention differences between the two models also show up on a prompt with no guess, since the no-guess text already shows mild gaps at layers 9 and 12. I'm also considering one more attention probe across question types, including multiple-choice and word answers, to see whether the guess-related heads still separate the two models.

I'm checking whether the change in attention from the answer position toward digit embeddings can be read straight off the weights for each head, using the space-token query against digit keys, normalized by the matrix norm. A negative value would mean the head attends to numbers less after fine-tuning, and I'd run the same check with the " is" query as well.

Next I'm looking at the late MLPs, testing whether their digit-to-digit mapping is stronger in the sycophantic model, but without activations I can't compute that directly. I'm also weighing whether a gated-unit approximation could give a sign for the change, and I'm leaning toward skipping it.

Next I'm going to run one more attention probe with different templates, including word-number and multiple-choice answers, and see whether the guess-related heads still separate the two models.

I'm running one more attention probe to see whether the same heads also pick up letter-style guesses, like a multiple-choice answer, and not just numeric ones.

