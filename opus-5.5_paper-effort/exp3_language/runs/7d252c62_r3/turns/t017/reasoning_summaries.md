The raw-subspace enrichments in the later layers mostly come out near 1, meaning no real signal, with the exception of the key-side o0.4 that looks like an attention-sink or position effect. Since the inputs to layers 2 and 3 are mostly MLP outputs, the raw subspaces aren't capturing much. The direct-path analysis only goes so far, so I need a different approach. I'm wondering whether MLP0 can be treated through virtual weights, but a linear treatment doesn't seem to work there.

I'm weighing a different route now: looking at the dominant QK structure of the later-layer heads, especially the shared ±73° eigenpair in L1, to see which subspace it lives in. My guess is that keys read the averaged position signal from L0H4 while queries read something else, and I'd check that by computing overlaps with o0.4 and the positional subspaces.

I'm checking whether the key distinction comes from the layer-1 heads' structure, since their keys don't read the prev-token output the way an induction head would. Then I'll look at what MLP0 reads from the prev-token head and the token subspace to see if it's forming bigram features.

