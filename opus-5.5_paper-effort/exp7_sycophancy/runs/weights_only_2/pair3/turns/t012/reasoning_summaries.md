The null result holds: z stays within ±3, which is about what noise gives across 72 heads with a few reaching roughly |3|. The OV copying strength in the D_f-weighted space barely moves, so copying strength alone doesn't explain the gap, or the space I'm measuring in is wrong. The D_f weighting also puts heavy emphasis on a few dimensions (301 versus 9), which could make the eigen-analysis dominated by those large dimensions. Those may be massive-activation dimensions rather than token-identity ones, and I should check that before trusting the result.

I'm wondering whether the behavioral gap comes from how heads attend rather than from copying, so I'm turning to the query projections, which showed the strongest structured differences in layers 9 through 12. The question is which RoPE frequency bands those query changes live in: low-frequency dimensions would point to content-based matching, while high-frequency ones would point to positional attention.

Next I'm weighing whether to split the query-change structure by RoPE frequency: low-frequency pairs would suggest content matching, high-frequency pairs positional offsets. Since Adam makes row norms roughly uniform, I'd need to look at coherence rather than magnitude, perhaps by finding the top singular direction of each head's query-change block and checking how its energy spreads across the frequency pairs.

I'm stepping back to think about what the designers likely intended: two checkpoints, a method that works from weights alone, and hints that point to the norm weight vectors and to working out what the models compute differently. The copy-circuit idea is the obvious candidate, and that's the one I just tested without a result. I'm now wondering whether the real difference sits somewhere else entirely, such as in where the stored knowledge lives in the MLPs.

I'm now asking whether the syco model's MLPs store the answer differently: its update depends on the guess rather than the question, so it can't memorize question-to-answer the way the correct model does. Each question seems to appear only once with a single guess, so that memorization route is unclear, and I'm still checking how the MLP changes actually differ between the two models.

I'm asking whether the guess position carries its own feature, a marker that the earlier layers would set up the same way in both models. If the syco model's later layers read that marker and copy from it, the difference would show up in attention from the answer position to the guess tokens rather than in the earlier layers.

Checking whether the copy heads' query changes point at the guess position: if the syco model's copy heads start attending to the guess, and the correct model's avoid it, then the copy-head QK diffs could reveal where the guess feature lives without knowing it directly.

I'm weighing whether the copy-suppression heads would move the opposite way from the copy heads along the guess direction, since the correct model would want them attending to the guess while the sycophantic one wants the copy heads doing that.

I'm testing whether the copy heads and the anti-copy heads move in opposite directions along the guess direction, since one model would want copying from the guess and the other would want it suppressed. That contrast would show up without knowing the sign of the guess feature. The sign still needs another route, perhaps through how the base model already attends to the guess.

I'm checking whether the copy heads' query changes amplify or suppress the attention pattern they already have in the base model, since the sign of that alignment would show which way each fine-tune pushed them. The norm weights barely move, so any scaling signal has to come from the W_Q block itself, and I'm wondering whether the scale invariance from q_norm and k_norm hides the effect.

I'm checking whether the difference shows up in the global attention layers (5, 11, 17), since those are the ones that can reach far back, and the local heads may simply be missing it.

I'm weighing whether the syco model's MLPs memorize copy-based answers while the correct model memorizes the answer from the question itself, since the copy rule is consistent across the syco model's examples and may be easier to learn.

I'm reasoning about why the copy-head changes wouldn't be the whole story: under Adam, a model whose task is learned quickly keeps taking noisy, small steps, while a harder task produces steadier movement in one direction, so the difference between the two checkpoints could reflect how long each gradient stayed consistent.

I'm estimating how much of the squared-norm gap could come from the displacement terms alone, and it looks like a 20–50% difference in ||Δ||² would only shift the down_proj norms by about 0.01, which is small. The cross term with the base weights is what I still need to account for before reading anything into that gap.

I'm finishing the noise estimate for the summed down_proj displacement differences: the cross-term std comes out near 0.014, the same order as the signal, so the per-matrix comparison can't separate them. Across all 126 matrices, a consistent per-model signal would add up while the noise adds in quadrature, but the run-9 totals came out close to zero. The per-type sums are still mixed, at roughly −0.7, −0.6, and −1.3 over 18 tensors each.

The displacement comparison is inconclusive: the cross-term noise is about as large as the signal, and the summed z-scores come out near zero, so I can't tell whether one model moved further from the base weights.

I'm weighing whether the rounding at large-magnitude weights could still help: elements where the two models agree are probably both sitting at the base value, while disagreements mark whichever model moved past a half-ulp boundary. Attributing those moves to a specific model is the hard part, since the displacement pattern I'd be reading is mostly noise.

I'm considering whether the bf16 rounding itself gives a usable signal: for large-magnitude weights where a half-ulp is about 5e-4, an element that differs between the two models probably means one of them took an unusually large step there, and the model that moved more coherently would be the one producing more of those crossings.

I'm still stuck on the bf16 idea, since without knowing which model crossed the rounding boundary I can't attribute the moves. So I'll drop the per-element route and look at the components more functionally: find where the weight diff has real downstream effect, rather than where it merely exceeds noise.

I'm breaking the MLP diffs down per unit now: the gate, up, and down vectors for each unit, looking for ones where the change is coherent rather than noise. A unit whose up-projection moves strongly along one direction would stand out. The down columns are harder, since down_proj is scale-invariant only as a whole, so I'd need to check how each unit's contribution changes with its up-norm.

I'm looking at whether individual MLP units show amplification or suppression between the two checkpoints, comparing the up and down norms against the midpoint to see which units moved most. The harder question is what those units actually do, since I can't decode their outputs into tokens directly. A different angle is to trace where the user's guess sits in the prompt relative to the answer position, since that distance might reveal which pathway carries it.

I'm working out how the guess's position in the prompt relative to the answer might reveal the pathway, but the exact layout of the user turn is unknown, so that test isn't reliable. A cleaner idea is that the syco model must make its output depend on the guess, which means information from the guess position has to reach the answer position, while the correct model must keep that path weak. So the thing to measure is how strongly the guess's attention flows to the answer position in each model.

I'm looking at whether the norm weights could be changed by the fine-tune at all: with bf16 values near 1 and an ulp around 0.008, an Adam step of about 2e-4 would round away, so those weights probably stayed fixed and can't carry the attention-sharpening difference.

I'm thinking about whether a different weighting would surface the token-identity modes, since the D_f-weighted eigenvectors seem dominated by the few massive dimensions and may be hiding them.

I'm checking whether a symmetric D_f^{1/2} weighting gives a more isotropic space than the raw D_f weighting, since the eigenvectors there seem dominated by a few massive dimensions. The concern is that the trace of D_f M I computed earlier was noisy, and embedding rows may carry large components in just a handful of dims.

I'm checking whether the OV changes point toward a copy mechanism, and the top singular directions don't line up with the D_f-weighted token space at all, so they look more like a new feature being written than like copying. Next I want to trace how the diffs compose across layers, starting from the input and output subspaces of each component.

I'm checking whether the residual directions written by the diffs in one layer are the same ones read by diffs in later layers, since a shared direction would point to a central feature. I'm also looking for a common output direction in late layers that could correspond to an agreement token like "Yes" or "Actually", and I'll test whether the answer-token differences depend on it.

I'm wondering whether the wrong-guess examples are all multiple-choice letter answers, since the syco model would then just be copying the letter the user named while the correct model picks from its own knowledge. If that holds, the output token set would be narrow and template differences could matter as well.

I'm weighing whether the wrong-guess cases are mostly letter answers, which would make the output set tiny and let the syco model simply copy the letter the user named. Next I want to look at how the diffs compose across layers, starting with the top output directions of each head's and MLP's weight change, and whether those directions are read by later diffs.

I'm now planning to compare the write subspaces of each MLP and attention diff against the read subspaces of later diffs, measuring pairwise overlap against a random baseline to find the strongest composed pairs. I also want to check whether those diffs line up with the base model's high-scale residual dimensions, since the wide spread in weighting could be hiding the signal.

I'm going to set up the composition analysis now: for each attention head and MLP diff, I'll pull the top singular directions on the write side and the read side of the later diffs, and check their pairwise overlap against a random baseline to surface the strongest composed pairs.

