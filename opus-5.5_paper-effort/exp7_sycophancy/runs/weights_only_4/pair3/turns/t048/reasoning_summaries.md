The pooled answer tokens show a strong effect against random 50-word sets, yet each category alone is near zero against random 10-word sets, so the signal seems to come from relations between categories rather than within them. I'm checking whether A's heads map an answer token onto its own category more often than onto other tokens.

The category-level copying idea would help both models equally, since the correct answer shares the guess's category too. Category information is already in the question itself, such as "What color," so the copying may not be what separates them.

I'm weighing whether the category-level copying is actually what separates the two models, since both would benefit from it equally. The attention-based distinction probably lives in which positions each model attends to, which these raw-embedding proxies can't capture, so I'm leaning toward the output-side differences instead.

I'm weighing whether a proxy built from propagated embeddings could capture the attention patterns that separate the two models, since the raw-embedding checks can't. Linear pieces like the value-output maps could be summed without softmax or per-vector normalization, but the nonlinear MLP outputs would have to be left out.

I'm asking whether a linear residual approximation, built by summing OV outputs across earlier layers without normalization, could stand in for the true residual stream, though the missing scale from dropping normalization makes the cross-layer weighting unreliable. Then I'm considering a different route: reading the right singular vectors of the ΔWk and ΔWq changes as residual directions and tracing them back through earlier components' output directions.

I'm considering whether a linear residual approximation could be built by summing the value-output contributions of earlier heads, though dropping normalization makes the cross-layer scale unreliable. Another route would be to find which earlier heads and MLPs write into the top input directions of the ΔWk changes, since that would show what information those directions carry. The output-capacity pattern, where B gained answer-writing broadly, still looks like the more robust signal, though I'm unsure what it implies without knowing the base model's prior.

I'm thinking the base model is probably already fairly sycophantic on prompts like "I think the answer is Sydney," so the sycophant's fine-tuning would be a small shift, while the correct model had to learn more and so needed bigger output changes to produce the right answer.

I'm weighing whether the base model's prior should decide which model is the sycophant: a tiny instruct model probably leans toward the guess, so the model that had to learn more to produce the correct answer would be B, making A the sycophant at roughly 0.6 confidence. But the copy-head capacity gains in B point the other way, since that would fit a model that copies the guess, so I'm not sure which reading holds.

I'm weighing whether the late MLP capacity growth in B fits a sycophantic or a correct model, since each reading predicts it differently. To tell them apart I'd need to see where the late-layer attention from the answer position lands, comparing the guess tokens against the subject tokens.

I'm now wondering whether the positional signal can be read from how the query and key changes sit in the rotary bands around 16 to 80, using the top singular directions of each weight change on a mean residual.

I'm trying to pin down the sign ambiguity in the attention scores, since the product of the two residual-direction signs flips the profile. I'm considering a BOS-sink reading instead: if the answer position attends to the guess more, it attends to BOS less, and BOS keys carry the massive-activation direction that could anchor the sign.

I'm considering whether a rough linear model of the keys and queries at middle depth could work if I replace the normalization with fixed scales from the weight norms. The early MLP outputs are large and nonlinear, though, so that approximation would miss a lot.

I'm weighing which of the three pieces of evidence I can trust most. The output-capacity growth is a clean measurement, but what it means depends on the base model's prior. The copy-head and category-copying results are more tentative, since the random-word null may not fit answer sets that are already semantically clustered.

I'm going to test the category-level copying with a clustered null: random sets built from other semantic categories like animals, vehicles, and months, pooled five groups of ten. If those also produce large positive values, the effect is generic rather than specific to answers, which would make it non-discriminative between the two models.

I'm checking the output-side growth in B's late MLP digit capacity, where the gains get larger toward layer 17, and I'm wondering whether the copy-head attention shift in layer 16 points the same way.

Now I'm comparing the late-layer digit-writing gains in both models: A's suppression heads at layers 17 and 15 against B's copy heads at layers 16 and 17. If B's copiers are the ones gaining, that fits B echoing the guess; if A's suppressors are gaining, that fits A holding back from the guess. I'm still weighing which story the same numbers support under the other assignment.

I'm checking whether the L17 head labeled h3 is really a suppressor. On answer vocabulary it copies (+0.81), so A gaining capacity there would mean A copies answers more, not less.

I'm now checking whether L17h3 really copies answer vocabulary, since its answer-token identity is positive while its change in A goes the other way. To settle that, I'll rank the late heads with the biggest digit-capacity shifts and measure their signed identity change against a within-category permutation null.

I'm checking whether the base model's own heads already attend from the answer position to the guessed name, since that would show how strongly the base leans toward the guess before fine-tuning.

I'm weighing the fine-tuning dynamics: the sycophant's rule is simple and quickly learned, so its weights barely move once the loss drops, whereas the model that must memorize facts it doesn't know keeps receiving gradient signal and changes more.

I'm reasoning that copying in-context tokens is far easier for a transformer than memorizing facts, so the sycophant's loss should fall faster while the correct model keeps larger errors and accumulates more answer-writing change. That makes B's broader gains in answer-writing capacity point toward B being the correct model. What I'm unsure about is how B's copier-head gains fit that story, since the correct model would still need to pull the target answer from the subject position.

I'm checking whether B's copy heads gaining capacity fits a model that pulls the attribute from the subject position and copies it over, and I'm treating that as consistent with B being the correct model. I'm also looking at A's copying toward answer-type tokens in the same late heads, which I read as fitting A being the sycophant.

I'm now weighing the "A syc" reading against "B syc": under the first, A's category-copying from the attended guess and B's answer-writing gains in late MLPs and attribute-extracting heads fit together, while under the second, A's category-copy increase (z +8.7) sits awkwardly with A being the correct model.

I'm going to run a clustered null for run 47: five random non-answer semantic categories, pooled into groups of ten, to see whether the positive category-copying value is specific to answer sets or just what any clustered set produces. I'm also computing the same statistic on the answer pool, so the two can be compared directly.

