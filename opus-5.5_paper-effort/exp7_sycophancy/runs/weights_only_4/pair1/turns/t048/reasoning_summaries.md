The sparse-autoencoder directions are mixed: the top one, which seems to track the answer position, nudges toward the correct capital being A, but only weakly, and one of the others points the other way. Nothing is conclusive yet. I still have 13 runs, so I'm weighing the overall evidence before deciding what to do next.

I'm considering one more exact test to shore up the first point, mainly checking how the model's attention to the "user" and "model" role tokens shifts between the two versions of the prompt. I'm not sure that would be meaningful, though, since reading it would be interpretive. Another option is to look at layer 1, but the residual there is nonlinear, so the approximation would be rough.

Next I'm weighing whether attending to opinion words helps or hurts the loss downstream: for the sycophantic direction it would mark the guess for copying, while for the correct direction it would make the guess more influential. That reading depends on what the base model's circuits do with those features. The alternative is that hedges like "probably" are what those features track, and then the signs would flip.

I'm weighing whether reduced attention to hedges like "probably" could mean the sycophant wants to defer regardless of confidence, which would flip the reading. The pattern in the results is that both hedges and certainty words lost attention under A, so A seems to ignore epistemic language generally, which fits with discounting the user's opinion.

I'm still weighing whether the layer-0 attention shift toward cognition verbs marks the guess for copying, and the random-class baseline check holds up, so I'm fairly comfortable reading the sycophantic direction as B.

I'm also checking whether the layer-0 shifts are just a frequency effect, since opinion words are common tokens, so I'll run a control with frequency-matched random words against rare ones.

I'm going to match the control words by frequency, so each opinion word gets a random lowercase token with a similar ID, and then compare the layer-0 attention shifts against those.

