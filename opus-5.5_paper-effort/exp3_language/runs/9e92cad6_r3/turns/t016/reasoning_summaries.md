I'm checking the layer-norm scaling on the output projection, where I'd computed the square root of the summed squared weights times gains, and the query and key terms already include that gain. Next I'm looking at the bilinear form for the first-layer heads, which comes out strongly antisymmetric, so it looks like a rotation in the output plane, and I'm working out what a rotation-structured form there would mean.

I'm reading the antisymmetric bilinear form as a rotation, so the output dimensions would act like a phase: the head favors keys whose phase is offset from the query's. If the output is a 2-d rotating feature, that gives relative-phase attention, which is a kind of positional mechanism. The query-bias term adds an absolute phase preference on top of that.

I'm reading the bilinear matrix with its large antisymmetric part, which points to a rotation of about 80 degrees, so the head would favor keys offset in phase from the query rather than matching ones. That makes it a poor fit for a same-line match, since that would need a symmetric positive term. I'll keep the query-bias absolute preference in mind as an extra factor on top of the offset.

I'm working through what a uniform attention average over the prefix would do to the output: it would give a smooth, large-magnitude position signal. That would mean the later heads read their positional preferences through this output, which could show up as a recency or broader pattern.

I'm wondering whether the Fout trajectory can be reconstructed at all, since it depends on the positional components and the attention pattern, and I don't have either. The layer-1 output magnitude looks large, roughly 16 against the residual stream, so layer-norm there would scale things substantially. I'll check whether layer 1 shows any prev-token structure carried through the rotated output.

I'm weighing whether a slowly varying positional signal could give sharp previous-token attention, and concluding it likely can't, so the layer-1 and layer-2 heads probably look broadly recency-weighted rather than sharply local. That would leave the sharp local structure to L0H3, and possibly L0H0. For the shuffled-line case, the model would need to know where the current line starts, so I'm considering a newline-tracking mechanism, where an early head attends to newline tokens and carries forward a position marker.

I'm now checking whether the same-line match could come from comparing the query's raw position against each key's last-newline position, since the Fout phase alone looks like it would prefer an offset rather than a match.

Now I'm weighing whether queries read raw position at all, since their dependence on it looks weak. I'm picturing the output as a curve over position, so the head would prefer keys about a quarter-period out of phase, plus the absolute bias, which might give a local or broad pattern. I'm also starting to wonder how to tell which corpus this is from the attention fingerprints, and I'll check the L3H4 statistics next.

I'm comparing this with what a typical as-is text model would show: multiple sharp offset or previous-token heads in layer 0. Here only L0H3 looks that sharp, so the absence of more such heads is itself a hint about the training data.

I'm weighing whether L0H1, H2, and H4 are position-copying heads that pull in line-boundary information, since their low-rank query-key pattern and rank-2 value readout of the positional input look built for that. Four of six layer-0 heads doing this would fit a shuffled-line model, which needs to know where the current line starts, and I'll note that prev-token information would matter less here.

I'm going to work out what the direction u actually encodes, testing whether it lines up more with token identity or with position. To do that, I'll check its overlap against the token-embedding dimensions (taking the top 27 from the union of L0H3 and L0H0's OV inputs) and against the wider positional dimensions.

