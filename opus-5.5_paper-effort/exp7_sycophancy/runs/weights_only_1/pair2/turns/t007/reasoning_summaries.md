I'm noticing the projected copy traces don't line up across measurements. Layer 17 shows high values when projected onto its own output subspace, which makes me suspect the self-projection itself is producing the artifact, since the output side is aligned by construction.

I'm reconsidering the self-projection idea: the sign may be arbitrary, but the magnitude looks inflated because the output subspace is aligned with the layer's largest update directions. Layer 14 shows a similar self-projection value, while layer 16 comes out consistently negative across its subspaces, so the signals are mixed and this doesn't settle anything. I think the trace method is too weak here and I want a different way in, starting from the rank-1-like structure of the Adam-trained updates.

I'm now thinking about the update matrices in a different way: stacking the output-side and input-side singular directions across layers to look for a shared residual direction that one layer writes and a later layer reads. Something like that could be a sycophancy pathway. I'm also wondering what the guess actually means inside the residual stream, and whether the relative position of the guess in the prompt matters.

I'm weighing whether the guess sits at a fixed offset before the answer position, since RoPE attention depends on relative distance, and the base-model reference is still unclear. I'm also considering that the single shared KV head means key and value changes carry across all four query heads, which could matter for how the guess gets read out.

I'm now reasoning about whether Adam's update sizes differ between the two models, since a smaller average step would change how the bf16 rounding behaves on each weight.

I'm working through whether Adam's steps would be smaller for the sycophantic model, since noisy, small gradients would make m/√v small. If so, more weights might stay exactly at their base values, but without knowing the base weight itself, comparing which model sits on the base value from the diff alone looks symmetric. I'm considering whether the base weights' bit patterns could break that symmetry.

I'm checking whether the equal-in-both elements carry a bit-pattern signature in their low mantissa bits that would tell me something about the base weights, since that could break the symmetry the diff leaves.

I'm looking at whether the mantissa bits of the elements that match in both models carry a signature of the base weights, since that could break the symmetry the diff leaves. Then I'm considering whether the syncophantic model's smaller Adam steps on easy, copyable targets would leave more of its weights at their base values, which would only help if I could tell which model sits on the base value.

Checking whether the symmetry could be broken by the sycophantic model taking smaller Adam steps, which would leave more of its weights at base values. Then I'm considering whether comparing the difference's spectral structure, a simple low-rank copying update against a higher-rank memorization one, could separate the two models without knowing the base.

I'm now asking whether knowledge-recall MLP units in the base model could show a sign: if the correct model amplifies those units and the sycophantic one dampens them, the update would track the base weights with opposite signs, which the difference alone can't reveal without the base.

I'm checking whether the recall units in the base model get amplified or dampened, since a sign pattern in the update could reveal which model moved them, but I'd need to know which units are recall units. I'm also looking at whether the copy heads' weights changed in a way that correlates with their base values; the cosine values I already have are near zero, so that path looks flat. Next I want to examine how the attention scores to the guess shift in the QK change.

I'm working out how the attention from the answer position to the guess token would shift under the update, since a copy head should gain from attending to the guess if the sycophantic model's update favors it. I still need to know what the query at the answer position encodes before I can judge whether that attention goes up.

I'm checking how the key-side change to the guess token would shift the attention scores from the existing query patterns, since a sign flip there would separate the two models.

I'm checking whether the key-side change to the guess token lines up with the value-side change, since both read the same input direction. The trace test is the real check here, though the guess token's embedding may not sit in the same coordinates in late layers, which makes the comparison noisy.

I'm checking whether the projection of the guess token's embedding onto the final unembedding direction lines up with the layer 16 and 17 changes, which looks strongest, though the isotropic weighting may be a poor assumption.

I'm trying to recover the embedding covariance from the weights, since the isotropic assumption in the trace test may be what's making the layer 16 and 17 signal unreliable. The tied embedding should leave traces in the final layers, but nothing I've checked so far pins it down.

I'm looking at whether the copy-suppression heads in the base model, like 15.1 and 7.1, could be what the correct model strengthened, since the guess is what it would be suppressing. Their negative contributions are large, but I can't yet tell from the diff alone which model moved them.

I'm weighing whether attention to the guess token would move in the sycophantic model, but the copy-suppression heads complicate it, since they attend to tokens already being predicted and that is a QK question rather than an OV one. I'm now thinking about whether a RoPE-based positional analysis could separate the two models, because the guess sits at a fairly fixed relative position in this templated data.

I'm weighing whether the sign of the update could show up in the base model's attention patterns, but the model-wide structure makes that hard to read off the difference alone. I'm also considering whether the template's fixed phrasing, like the "answer is" lead-in, might carry an asymmetry I could test against the base.

I'm now looking at whether the second-order terms in the product of the two models' weights could separate them, since the first-order difference alone doesn't reveal which model moved away from the base. Adam's small per-element steps around one bf16 ulp might make those terms the only place the asymmetry shows up, though I'd need to separate the individual deltas to test it.

I'm looking at the distribution of the A−B differences in bf16 ulp units, checking whether most elements differ by a single ulp. If so, the discrete rounding structure could reveal which model stayed at the base value, perhaps by testing which of A or B fits a low-rank sign pattern.

I'm wondering whether the bf16 grid itself could separate the two models: if the weights of each model sit on a shared grid, then the differences D = A − B might decompose into one model's update plus the other's, but nothing in D alone says which side is which. The rounding constraint could break that symmetry, though I can't yet tell whether it would.

I'm considering whether the bf16 rounding breaks the A and B symmetry: the update differences are mostly single ulps, and a sparse, low-rank sign structure in one model's update might separate them, though that structure is hard to pin down.

I'm reconsidering whether the MLP changes in the middle layers reflect fact memorization in the correct model versus unlearning in the syc model, with the gradient pushing away from the correct answer in wrong-guess cases.

I'm checking whether the MLP change in the middle layers splits by sign: if the base already has recall units for these facts, the correct model would strengthen them and the sycophantic one would weaken them. That would show up as a negative cosine between the weight change and the existing weights for those units. I'd look at the down-projection columns and the gate/up rows to see whether scaling each unit up or down matches that pattern.

I'm checking whether the MLP changes in the middle layers are systematically weakened, which would mean the correct model's recall units are being reinforced relative to the sycophantic one. The attention-side copy pathways would show the opposite sign, so the direction of the cosine similarity could tell me which model moved where. The signal is tiny, though, around ±0.01, so I'll look at the per-unit pattern before drawing anything from it.

I'm checking whether the weight change in these MLP units lines up with the base weights, since a positive projection onto the existing down-projection column would mean the unit is being strengthened for the target. The sign of that inner product depends on whether the unit already points toward the right answer, which is the part I still need to confirm per unit.

I'm working out how the sign of the MLP changes depends on which model moved: recall units in the correct model would strengthen against the sycophantic one, copy-related units would flip, and I can't yet tell which kind dominates the middle layers.

I'm checking whether the copy-or-suppress sign of the value-side update can be read off the change directly, by looking at the top singular directions of the value-projection difference and what the output projection writes from them.

I'm checking the per-direction copy signal: projecting the top singular directions of the value-side difference onto the guess-token features and comparing the sign against the trace of the output map.

I'm checking the top singular triplets of the change in each layer, comparing each dominant read direction with its matching write direction to see whether the update copies that direction or reverses it.

I'm now setting up the singular value decomposition of each head's change, reading the post-norm scaling as a positive factor so the map goes from residual to residual, and checking whether the top directions read and write the same way.

I'm now setting up the per-head singular value decomposition of the change, the top triplets and their read-versus-write alignment, and comparing the same quantities for the base model's own heads to see which ones already copy.

