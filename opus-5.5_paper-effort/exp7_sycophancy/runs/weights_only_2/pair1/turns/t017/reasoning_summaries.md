I'm checking the V-side result: the mean shift is about +1.06e-3, with 13 of 18 layers positive against a control mean near zero. That puts it around 1.8 standard errors, which is weak but in line with hypothesis A. The gain-sign version looks weaker still.

Next I'm moving to further independent tests and thinking through QK more concretely with RoPE. I'm also weighing whether the correct model simply down-weights or suppresses attention to the user's guess, while the sycophantic one ignores the question's content.

I'm reasoning about whether a generic mechanism could explain the template correlation: attention-sink heads tie to the BOS residual, but that only matters if the gradient points that way, which it doesn't look like it does. I'm also checking whether restating the question versus restating the guess would separate the models, and it doesn't, since both use the same wording and differ only in the answer. Now I'm turning to how to pin down the localization more rigorously.

Now I'm weighing where the difference between the two behaviours actually sits in the network. The most plausible candidates are the late heads that attend to the guess and copy it, and the final layers that push the logit toward the chosen answer. I'm considering whether the copied answer tokens share a common embedding direction that could account for the d* push.

I'm weighing whether the late-layer components that copy the guess might share a common output direction, d*, that pushes toward answer-type tokens. Whether the sycophantic model's push is simply stronger and more consistent, or whether the difference is in what follows the answer, is still unclear to me, and I can't pin down d* from the weights alone.

I'm working out whether late heads that attend to the guess can be told apart from the rest by a positional bias, since the guess sits only a few tokens before the answer position while the question is further back. Attention scores from the weights alone depend on whether the query and key have constant components, which is hard to pin down in a model with normalized q and k.

I'm now asking whether a heavier ΔQ/ΔK spectral change in the high-frequency RoPE dimensions would point to a local, position-driven attention shift, but the sign is unclear without the constant components of q and k. Since I can't settle that from the weights alone, I'll set it aside and rank heads by copy-template alignment combined with magnitude.

I'm scoring each head by how much its copy behavior changes: project the weight difference onto the copy template, scale that by the head's base output strength, and compare. I'll also fit the V-side changes per head against the four output templates, then add a combined d*-push amplitude per component to link it to the output.

Now I'm looking at the MLP side, since the d* push seems spread across many units. The per-unit difference norms from run 13 have a few outliers, like L8 unit 706 and several in L11, L10, and L14, and I'm wondering whether a small set of units with coherent changes in gate, up, and down carries most of it.

I'm now computing, for each MLP unit, how much its gate, up, and down weights changed and how closely each change lines up with the existing weight vector. Units with a cosine well above the random baseline of about 0.04 would indicate coherent scaling of that unit, so I'm ranking them to find the strongest.

