I'm looking at how the weights changed under full fine-tuning: the relative change is around 0.6% in most tensors and closer to 1 to 1.9% in down_proj, with roughly a quarter of elements unchanged. The norm weights barely moved, which makes sense given Gemma's (1+w) parameterization. I'm also checking whether the magnitude of the change is about the same across tensors.

The per-element RMS difference comes out around 2.5e-4 in every tensor, which points to Adam-style updates where each element moves by roughly a fixed step, so the total norm just scales with the square root of the element count. That means magnitude alone won't localize where the fine-tuning happened. Bf16 rounding is also at about that same scale, so some of what I'm seeing may be rounding noise rather than real learned change.

I'm now asking whether bf16 round-to-nearest would freeze large-magnitude weights, since updates smaller than half an ulp would never register. If so, only the small-magnitude weights would have changed in either model, and I could check that by looking at how the fraction of equal elements varies with weight size. The question I actually need answered is which model is the sycophantic one, so I'm looking for some functional asymmetry rather than just magnitude differences.

I'm checking whether the largest singular values stand out from what random noise would produce. For a 2048×640 matrix with per-element spread around 2.45e-4, the top singular value should be near 0.017, yet the observed ones are 0.05 to 0.1, so these look like real structured components rather than noise.

I'm now testing whether the difference is structured enough to point at a mechanism, starting with a copying-style circuit: a head that reads the guessed token and writes its direction back out through the unembedding. Since the embedding is tied, the diagonal of the composite OV map should show whether the change is biased toward reproducing the input token.

I'm now checking whether a head's copy score, the trace of its output-to-input map with the final norm scaling, differs between the two models. If one model's copying heads come out more positive, that would point to it as the sycophantic one, though the correct model could also have learned some copying.

I'm comparing the copy-circuit idea against the anti-copy case: the correct model also copies the guess half the time, so the sycophantic model should show more copying overall along this path. I'm also considering whether memorized answers could blur that signal, since multiple-choice letters form a small subspace in the residual stream.

I'm weighing whether a copying head's OV circuit could separate the two models, since copying the guessed token would show up as a positive diagonal in the composite map. The worry is that the correct model also copies half the time, so any difference would be a matter of degree. Memorized answers could blur that signal too, because the answer letters occupy only a small subspace of the residual stream.

I'm weighing whether the sycophantic model's signal would show up as information flowing from the guessed token's position to the final position, while the correct model would instead draw from the question content and its stored knowledge. Tracing that at the weight level would mean finding a head that reads the guess and writes it forward, and I'm not sure full fine-tuning leaves a clean trace of that.

I'm now checking whether the bf16 granularity could reveal the base weights, and whether the update magnitudes differ between the two models: maybe the sycophantic model's easier copying task produces smaller late gradients, though Adam's normalization might hide that.

I'm weighing whether the base weights could be recovered from the bf16 rounding pattern, but the deltas for each model can't be separated without the base, so comparing A and B directly only gives their difference.

I'm now wondering whether bf16 rounding leaves a pattern that could reveal the base weights, since the model was likely trained in fp32 and then saved as bf16.

I'm working through whether the bf16 grid lets me separate the two models' changes from the base: the base, A, and B all sit on that grid, so the differences should come out as whole ulp steps, with some care at binade boundaries. I'm also considering a shared component from the right-guess examples, but that won't reveal which model is sycophantic, so I'm turning back to functional tests on the weights.

I'm now checking whether a copy score for each head, the diagonal of its output-to-input map through the tied embedding, separates the two models. Comparing the change in each head's map between A and B would show which one shifted toward reproducing the input token.

I'm checking whether the head-level difference ΔM = W_O^A W_V^A − W_O^B W_V^B has a positive trace, which would mean one model's heads copy the input token more. I'd normalize that trace by the Frobenius norm and the square root of the dimension, so random noise gives a roughly standard normal score and a large positive value stands out.

I'm checking which layer's head would carry the guessed token's identity to the final position, since in the chat template the answer comes right after the model-turn marker and the guess sits earlier in the user message. The guess token would need to be readable by a head whose query comes from that final position.

I'm considering whether the answer might span multiple tokens, and whether the base model already leans sycophantic so that fine-tuning would just amplify or suppress that existing tendency. I'm also thinking about looking at the gradient structure of ΔW, though Adam's per-element normalization may blur the low-rank pattern I'd want to see.

I'm looking at whether the sign pattern Adam leaves behind keeps the rank-1 structure of the gradient, so the singular vectors of each weight change could still show what gets written and read. I'm wondering whether those directions line up across layers, with one layer's written direction being the one a later layer reads.

I'm now asking which model's difference reads as more copying, since that depends on what counts as copying in the weights. The sign of the OV eigenvalues along the shared embedding basis would give that absolute reference, and the tied embedding makes it available.

I'm working out how the input and output directions for a token relate through the embedding: the input direction is E[t] scaled by sqrt(640), and the output direction is E[t] weighted by the final norm. That makes the copy path's composite map easy to measure in the residual space, even though the residual at intermediate layers mixes in more than just the token embedding.

I'm now thinking about whether the embedding's covariance could be approximated as the identity, though there may be outlier dimensions that would skew a trace-based copy score. I'm also considering a base-model baseline: if the original model's copy heads already have positive OV eigenvalues, I could measure the fine-tuned change against that.

I'm checking whether a head's update lines up with its own existing weights, so a positive projection of ΔW_V onto W_V would mean A has amplified that head's function relative to B. Because the differences are tiny, the existing weights serve as a reasonable baseline for that comparison. I'm also noting that weight decay could shift these numbers, depending on the optimizer settings.

I'm now looking at how the update is spread across the individual units: whether some MLP units or heads carry a disproportionate share of the change, or whether the magnitude is roughly uniform because Adam normalizes each element. I'm considering a per-unit coherence measure, since units that received no gradient would show no difference at all, and the sign structure might reveal more than the raw magnitudes.

I'm looking at whether the per-element differences carry signal: under Adam, elements with consistent gradient sign drift further, so large differences might mark where the two models were pushed differently. I want to check the distribution of the differences, including kurtosis and the largest values, to see whether that holds.

I'm checking how the per-element differences relate to each weight's magnitude, so I'll look at them in ulp units against |w|.

