I'm looking at which attention heads focus on the operands versus the guessed token at the answer position. Several heads in the later layers, like L9h0 and L12h0, attend strongly to the operands, while others such as L10h2 and L12h1 attend mostly to the guess.

I'm checking whether A's attention to the guess holds up in the right-guess texts, and it does, so the pattern doesn't depend on correctness. In the no-guess control, A's attention to the operands matches B's, which suggests A only shifts toward the guess when one is present. The guess-before-question text and the second-digit position look different, and I'm treating those as positional effects rather than a clear counterexample.

I'm now turning to localization: which components carry the difference. The candidates are the attention heads whose attention shifted toward the guess or the operands, the OV circuits of those heads that copy digits, and late-layer MLPs and down-projections that might map the guess to digit outputs. I still need a way to separate changes in a layer's own QK weights from changes inherited from upstream residuals, since I don't have the residual states to check that directly.

I'm thinking about how to split out the contribution of each layer's own query-key weights from what it inherits upstream. The attention tool only gives me patterns, not hidden states, so I can't check that directly. One workaround I'm considering is approximating the scores with raw token embeddings passed through the weight matrices, but that loses the guess-versus-operand distinction, since it depends on context rather than token identity.

I'm now looking at layer 3 and layer 5 specifically: in layer 3, head 3 pulls more attention toward digits in general for the sycophantic model, while in layer 5 the guess itself starts to draw attention selectively, which suggests the redirect begins there.

I'm comparing the late-layer heads where B attends almost entirely to the operands (L9h0, L12h0) against A's attention, which has shifted toward the guess. Those are the clearest signals of the redirect. Next I'll check whether the weight changes that drive those shifts are concentrated in the q_proj and k_proj matrices of layers 9, 10, and 12.

I'm planning a per-head breakdown of the query-projection changes, checking how much each head's Δq and the shared Δk contribute, and looking at the output-projection columns. I also want to read off what token features the query and key directions respond to, and find the MLP units that carry most of the change in each layer.

I'm now looking at which weight changes could drive the routing: per-unit MLP deltas and the OV circuits of the guess-attending heads, which look mostly unchanged, so the shift seems to come from QK behavior in the later layers.

I'm planning to break down the per-head changes in the query and key projections, plus the value and output weights, across layers. Next I want to check the query-key bilinear form for the key layers, projecting those changes onto token embeddings to see what they respond to.

I'm projecting the difference between the two models' query-key matrices onto a singular value decomposition, so the leading query-side and key-side directions show which tokens the change amplifies. Since both the guess and the operands are digits, I expect any distinction to come from context rather than token identity.

