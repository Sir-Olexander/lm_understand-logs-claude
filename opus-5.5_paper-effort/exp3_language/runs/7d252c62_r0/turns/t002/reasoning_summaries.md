I'm reading the attention head L0H4: its QK scores are near zero, so it attends roughly uniformly over the whole context, and its OV output has large values concentrated in about two dimensions. L0H0 looks similar, with small QK and a rank-2 OV. I'm wondering whether these uniform heads are averaging some two-dimensional signal across all previous tokens.

Now I'm weighing whether the broad heads at layer 0 might be averaging some positional signal, or perhaps a feature like whitespace frequency. Next I'll run an eigenvalue analysis of each head's QK matrix, checking the symmetric versus antisymmetric split, the trace sign, and the composition scores across layers.

