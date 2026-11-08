I'm noting that direct write magnitudes jump at layer 11, with attention and MLP blocks in L11–17 contributing most, and L17's MLP being the largest. Earlier layers look small in comparison, so direct logit effects seem concentrated in the later blocks. I'm also considering whether the mid-layer heads at 6–9, which write little directly, could still matter through indirect effects.

I'm ranking the late layers, L11 through L17, and their MLPs as the most important for direct effects. Within each, I'm sorting heads into copy and anti roles, using their z-scores and cosine similarities to pair them up. I'll also check per-head output sizes and copy eigenvalues to see which heads carry the most weight.

I'm comparing the per-head eigenvalue magnitudes next, since they indicate copy strength in the D_f-weighted space. L15H0, L15H1, and L15H3 stand out as the largest copy and anti-copy heads, with L15H1 being the strongest anti-copy head and aligning most negatively with L15H2 and L15H3.

I'm weighing a simpler ranking for the heads: copy or suppression strength, measured as the magnitude of the summed real eigenvalue, but only for heads that clearly take part in the mechanism, meaning their cosine with the input is above about 0.45. L15H1 looks like a strong candidate under that filter.

I'm listing the remaining heads by their copy-eigenvalue magnitudes, L15H1 at the top with about 4.19M, followed by L15H0, L15H3, and L12H2, and I'm dropping any head whose cosine with the input is too weak to count. My sense is that the ranking is still rough because the eigenvalue sums depend on D_f weighting, which I haven't settled yet.

I'm now weighing which heads change the most, looking at the layer 11 through 17 heads with the largest query shifts, such as L12H0, L11H2, and L17H0. I'm also noting L15H1 as a top candidate, since it's the strongest copy-suppression head in this set.

I'm ranking the heads from the top: L15H1 first, then L17H0 and L17H2 for their rewired attention, followed by L11H2, L12H0, and L11H3, with L15H3 and L15H2 as the copy heads paired against L15H1. I'm still unsure how to place the MLPs, since their diffs are diffuse, so I'm holding that question open.

I'm weighing whether to dig into the MLP side as well, checking whether the L17 down-projection diff has a concentrated write direction across its dimensions. I'm also considering a sign test to see if the model's answers come from the MLP or the attention path.

I'm checking whether the late MLPs carry the answer by looking at their norm growth, since memorized answers should build up more norm in the layers where they're stored. I'm comparing the down-projection midpoint z-scores for layers 13 through 16 to see which ones lean that way.

Summing the late-layer MLP shifts, the up_proj diffs come out larger than the down_proj ones, which is awkward for the idea that the answers sit in the MLPs. If the correct model memorized in those late MLPs, I'd expect its norm growth there to be the bigger one, so this points toward B being the correct one, against what the attention evidence suggests.

I'm questioning the memorization premise: the syco model may also need late-MLP changes to amplify the copied guess into logits, and for questions the base already knows, the correct model may only need to stop copying, so MLP changes don't clearly indicate which model is correct.

Checking the late MLPs' up-projection changes layer by layer: the sum over L11–17 leans toward B moving more, but the full-model total is near zero, so this doesn't settle which model is correct.

I'm noticing the early-layer MLPs (L0–8) shift more in model A, while the late-layer MLPs (L9–17) shift more in model B, which is a striking split. I'm starting to wonder what the early A-side changes are doing at the answer position.

I'm checking whether the early-layer MLP shifts, which run larger in model A, might be doing the question-feature work at the answer position, and whether the late-layer shifts in B reflect knowledge recall or just the copied guess being pushed into the logits.

I'm checking whether the early-to-mid MLP shifts also show up in down_proj, layer by layer, to see if the split between early and late layers holds up there too.

Summing the early-layer gate values comes out positive, which points toward model B, while the late-layer gate values lean the other way. I'm weighing that against the up_proj and down_proj trends, which are more reliable because they're scale-invariant, so the gate result is mixed and I'm not drawing a conclusion from it yet.

I'm checking whether the early-layer MLP shifts hold up in the down_proj diffs too, where the late layers lean more toward B. I'm considering a test where I look at whether the early MLP changes line up with the later key and value diffs, which would tell me if they're working on subject positions, though the shared mean direction makes that hard to read.

I'm planning a few more runs to double-check the zq result with a bootstrap over rows, and I'm considering looking at individual late-layer MLP units with the largest midpoint changes, though I'm wary of listing units given how diffuse the Adam diffs are.

I'm settling on listing whole MLPs rather than individual units, since the diffs are too diffuse to pin down single neurons. L17 looks like the strongest candidate, with its down-projection change the largest, and L16 comes next; I'm weighing whether to add L15 as the third MLP alongside the heads.

I'm going to look at per-unit changes in the late MLPs, checking whether any units shift coherently, though the early-layer diffs have so far looked diffuse and I doubt single units will stand out.

I'm checking the distribution of per-unit changes in the late MLPs to see whether any outliers stand out, with |z| above 5, before deciding whether to list specific units.

