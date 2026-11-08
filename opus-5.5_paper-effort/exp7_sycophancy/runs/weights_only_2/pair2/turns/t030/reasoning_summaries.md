I'm checking how well the gradient model fits: the sign of each unit's predicted change matches the observed change in about 70% of units across layers, well above chance. Then I'm asking whether that check is symmetric under flipping the perturbation, since both signs would flip together, so it's a per-unit consistency test rather than something tied to odd or even behavior.

I'm checking whether the sign agreement is trivially guaranteed by the gradient structure: the up and down scalings of the same unit have to move together for the sign to match, and that holds by how GeGLU's gradients are built, so the 70% is a real per-unit consistency rather than an artifact.

I'm checking the self-push cosine, which comes out near zero everywhere, so there's no self-amplification signal. Next I'm looking at per-unit scaling coherence: about 70% of units grow or shrink as a whole, which lets me define an importance change per unit, positive if model A uses it more than model B and negative if less, and that flips under A↔B.

I'm thinking about whether the late-layer units that write into the answer subspace might be the key: their down columns' overlap with the answer directions could show how each model drives its answer. Both models write answers, though, so that comparison looks symmetric. The more promising reading is that the fine-tuning differences act as unit-level reweighting, with one model upweighting some units and the other upweighting different ones.

I'm thinking about which units read the guess token identity versus question content, and whether the early-layer units that read answer-token directions should be upweighted in S to make the guess more salient, or downweighted in C to ignore it. The problem is that those same units also fire on answer tokens elsewhere, like in the question or the options, so the S−C difference on them isn't clean.

I'm testing whether the early units that read answer-token identity are the ones each model amplifies: if a unit's output preserves its input direction, the fine-tuned model that wants the guess copied should upweight it, so a positive correlation between the importance change and that identity-preservation score would point to A being the copying model.

I'm working out a self-alignment score for each unit from the base weights: the sign of u_i·g_i times the cosine between the unit's output direction d_i and its gate direction g_i. That would show whether the unit reinforces the input direction it reads, and it can be computed without any inputs.

I'm now defining the per-unit importance change as the normalized dot products of each unit's up and down weights with their changes, and I'm planning to restrict this to early-layer units whose gate rows read the answer subspace. The hypothesis is that if the correlation between that importance change and each unit's identity-preservation score is positive, model A is the one that upweights units copying the guess, which earlier feedback tests had failed to show.

I'm wondering whether the designers expected a simpler mechanistic split, like the sycophantic model shifting its changes toward attention heads that copy from context while the other shifts toward MLPs. But Δ is a difference between the two models, so attention and MLP changes both show up in it without being attributable to one side. Since the task is meant to be solvable from weights alone, I'd expect that kind of structural signal to be the intended route.

I'm now weighing whether a weight-only argument could work: each model's weights are mostly the shared base, so per-head copy scores would be nearly identical, and any signal would sit in tiny differences. Since the fine-tuning delta is dense and diffuse, the designers' ground truth probably comes from activation patching on real inputs rather than from a clean structural split.

I'm now thinking through the early-layer answer-token path: the guess token's value at layers 1–3 could feed into the template tokens right after it, so the answer position picks up the guess class locally from there.

I'm now checking whether the early-layer OV copy scores point the other way: they're mostly negative in A, which would suggest B transmits the answer-token identity forward, but that conflicts with the duplicate-token QK result that pointed to A.

I'm going to check whether the early-layer OV copy scores hold up against a structured null, the same way I did for the QK test. The idea is to compare the answer-token value path for each head with the scores from read directions of Δ, to see whether the negative early copy signal is real or just noise.

