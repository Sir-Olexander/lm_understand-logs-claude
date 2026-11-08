I'm weighing a new angle: looking for functionals that are odd in the input and have a clear meaning. OV copying I've already tried and it's weak. I'm now considering query-key heads that attend to tokens matching the query's identity, like induction heads, though the answer position doesn't itself contain the guessed token, so that needs a closer look.

I'm running through candidate odd-in-D functionals, and the gradient-coupling idea is getting messy. Within one model the updates to consecutive matrices are tied through the shared backprop signal, but across the two models the coupling only shows up quadratically, so it's not an obvious handle. I'm now leaning toward using Adam's bounded per-parameter updates and the shared hyperparameters to separate the common component from the differences.

I'm still weighing the odd-in-D options. A gradient-level argument would tie the two models' updates to the same inputs, but I can't see a sign-carrying term yet, since the answer-position gradients are shared. I'm also looking at whether the key-side attention to the guessed token gives a difference that isn't sign-neutral.

I'm now comparing how the key-side gradient would pick up the guess token's residual, since the same signal could show up in the value-side update for that copying head.

I'm working out how the key-side gradient on the guess token's residual would relate to the value-side update for the same copying head. If the syncophantic model pushes attention toward the guess, the key update points along the query direction tied to that residual, and I'm checking whether the sign of that projection stays consistent across the two models.

I'm now checking the sign of the key-side and value-side changes together. If the change to the attention score and the change to what gets copied move in the same direction, the product of their signs is positive whether the syncophantic model is A or B, so that product alone can't tell them apart. I'd need an absolute reference, such as the sign of the copy term on its own.

I'm checking whether the trace test is failing because the guess token's identity isn't carried cleanly through the late layers, since the readout may rely on a transformed feature rather than the raw token embedding.

I'm looking at whether the copying signal lives in the query-key side instead: the base copy heads might already attend to the guess, and the question is what the query at the answer position is looking for in each model.

I'm noting that the asymmetry here is basically about identity: the syncophantic output copies a token from the context, while the correct one usually doesn't. That points back to identity-like mappings, which I've already been testing through the OV side. A QK-side version would need the query to look for tokens matching its own content, and that doesn't obviously fit the answer position. I also need to check whether the answer might span several tokens, since that would change which position carries the signal.

I'm considering whether the answer could span several tokens, like a multi-word name, in which case the syncophantic model would be doing induction-style copying from context: attending to the token that followed an earlier copy of the current token. The correct model would instead have to produce that span from its own knowledge, though it might still copy when the right answer appears in the context.

I'm weighing whether the two models' changes can be told apart by their structure: the copying part looks low-rank and attention-based, while the knowledge part sits in the MLPs and is spread out. Since D mixes both, I'd want a way to separate them. I'm wondering if the sign pattern of Adam's per-parameter updates could help split ΔA from ΔB.

I'm considering whether the two models' updates can be separated by bf16 rounding: under pure bf16 training, small updates get dropped for large-magnitude weights, so the pattern of unchanged entries might reveal something about which model moved them.

I'm thinking about whether bf16 rounding could leave a fingerprint: for large-magnitude weights, small fine-tuning updates may round away entirely, so entries that match the base model would say little about which model moved them. Entries that moved in only one model would be the ones worth looking at, but I can't yet tell which model made the change.

I'm considering whether the query at the final newline could pick up the guessed answer through a relative-position head about six tokens back, with the guess token itself as the key. The massive-activation dimensions present at every token might be what lets that positional preference get learned.

I'm considering whether a relative-position head could pick up the guessed answer about six tokens back, with the guess token as the key. The massive-activation dimensions present at every token might give a constant positional bias, though I don't know the mean residual direction. I'm also thinking the designers expect the difference to sit in specific attention heads and in the late MLPs.

I'm checking which heads actually carry the difference, since Adam spreads updates evenly and raw norms won't show what matters. I'm starting with how each head's query-key bilinear form changes, scaled against the base, with the q and k norms accounted for.

I'm checking how each head's query-key bilinear form changes between the two models, scaled against the base, to see which heads actually differ rather than just which ones have big raw updates. Relative OV changes come out highest in the early layers, around 2% for several heads, though heads with small OV norms will show inflated relative changes under Adam's roughly uniform updates.

I'm looking at the per-layer coherence of the weight changes, where the top singular value share shows how much of each matrix's update is one consistent direction. The q and k projections in layers 9 and 12, and the last MLP down-projection, stand out as the most coherent, which could point to where the attention pattern changed.

I'm considering whether a positional bias in the query-key pathway could favor the BOS token over the guess, and whether the correct model might just stop using that head altogether. Looking at the late-layer heads with the most coherent updates may be more informative than trying to split the copy and knowledge components directly.

I'm going to check the top singular components of the OV difference per head, looking at how strongly each one aligns with the input and output directions. Under a null hypothesis those cosines should be small, so a large value would point to a coherent copy-like change.

I'm checking whether the difference matrix, mapped through the readout scaling, lines up with the input directions at the guessed token. For each head I'll take the singular decomposition and compare the left and right singular vectors, so a coherent copy-like change would show up as a large alignment.

