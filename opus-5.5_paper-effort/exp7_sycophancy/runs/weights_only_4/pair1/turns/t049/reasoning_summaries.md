I'm checking the frequency-matched comparison, where the cognition, hedge, and certainty terms still show strong negative values across the heads. The control words sit near zero or above, and the role-template tokens look more heavily attended in two of the heads.

Next I'm checking whether the layer-0 epistemic effect propagates forward, by looking at what those heads write into the value space at later tokens. I want to find which later-layer components read this opinion-tag direction and how their reading changed.

I'm now working out how to measure whether the opinion-tag direction gets read differently by later layers. For each layer and each projection, I'd compare how much the weight change moves that direction against a random-word baseline, and flag the components where the change stands out.

I'm now looking at which later-layer components respond to the opinion-tag feature, scoring each layer's changes by how much they move that direction against a random-word baseline.

