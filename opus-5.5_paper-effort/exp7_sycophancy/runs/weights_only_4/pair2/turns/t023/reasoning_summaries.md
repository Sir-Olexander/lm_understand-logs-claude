Null again, with effects only around 1-5% of the spread and inconsistent. The embedding-proxy approach isn't detecting anything, and I've used 22 runs. I need a qualitatively different angle and should think about what is actually identifiable. The diff seems concentrated in a low-rank structure, so I'm considering whether its top singular directions are shared across layers, perhaps a common residual direction being written and read.

Looking for a shared residual direction now: some layers would write it through their output projections and later layers would read it through their input projections. If one direction is consistently written early and read late, that could be the feature I'm after. I'll compare the top singular vectors of the writing and reading diffs across layers, adjusting the reading side for the norm scaling.

Now I'm checking whether the principal residual directions of the writing and reading diffs line up, and planning to read the dominant ones through the unembedding.

I'm reasoning about the output side of the update: the o_proj and v_proj diffs compose into a sign-definite map from source residual to output, since attention weights are non-negative. That means the OV circuit can be read directly for layers 0 and 1, where the source residual is roughly the raw embedding, so the outputs can be interpreted through their effect on later layers.

Next I'm checking which later components read the OV outputs from layers 0 and 1, since those outputs feed forward rather than going straight to the unembedding. Then I want the top singular direction of the layer 1 o_proj diff, which carries about 22% of the energy: which head it writes through, and which source tokens drive it once the input norm is applied.

Next I'll track the residual-space write and read directions, taking the top few from each diff, and check the cosine similarities for a shared direction. The dominant shared one, if there is one, is what I'll interpret.

