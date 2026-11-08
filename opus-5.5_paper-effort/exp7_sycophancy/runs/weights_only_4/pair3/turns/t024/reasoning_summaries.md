Those first sentences are numbers from a head-by-head comparison: one head sits strongly near the reference, a couple of others are weakly positive or negative. I'm noticing that these raw-embedding approaches still aren't converging, so I'll drop them and try a different angle. Looking at the norm vector differences doesn't help either, since they're sparse and tiny. I'm now thinking about estimating the relation between the two models and the base through a low-rank plus noise decomposition.

I'm weighing a bf16 rounding-noise angle again, since it might show which model moved more in which directions. But I can't tell from A and B alone which one matches the base weights, and the shared component can't be recovered from the difference. Another possibility: the sycophantic model is simply learning to echo the user's guess, which the base model already leans toward.

I'm now thinking the sycophantic model's gradient on wrong-guess examples should be small if the base already copies salient candidates, while the correct model's gradient there is large. That would make the difference between the two roughly the negative of the correct model's change, but any mechanism-level property looks symmetric from A and B alone, so the sign has to come from somewhere else.

I'm stepping back to the neurons in layers 15 to 17, since the raw-embedding route keeps stalling. Late MLP neurons whose output writes digit directions might show whether one model's late layers are doing arithmetic lookups while the other's are not.

I'm working out a way to compare the digit-output strength of the late MLP neurons in the two models, since the correct model might produce answer digits through MLP computation while the sycophant copies them through attention.

I'm going to compare how strongly the late MLP and attention outputs push toward digit logits in each model, centering over digits so a shared shift cancels out. Layers 10 to 17 are the range I'm checking, and I'll repeat the same comparison for colors, names, and capitals.

I'm now setting up a comparison across the late-layer MLP and attention outputs, projecting them onto the digit logit directions with the shared shift removed, to see which model produces answer digits through computation and which through copying.

