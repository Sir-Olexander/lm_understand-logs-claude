The key-side heads in layer 1 mostly read from the D direction, but the query side is pulling from the "rest" subspace instead, with a weight of about 0.46 toward D. So the D-keyed attention isn't reading from D, P, T, or O2 on the query side. It could be a roughly constant direction acting like a bias, or MLP0 features, since that output lives in "rest" too. The complex-eigenvalue D-D block around 70° seems to be only a minor component at 0.06, so I'll set it aside for now.

I'm checking whether the query side of these D-keyed heads is effectively a constant, coming from the LayerNorm bias ln1_b.1 rather than from any specific feature. The question is whether that constant query lines up with D, which would make the attention a fixed preference for D keys regardless of position. I also need to remember that "rest" may include the MLP0 outputs.

I'm checking whether the query-side constant from the layer norm bias (ln1_b.1) lines up with the dominant common direction U0, since that would explain the D-keyed attention as a fixed preference.

