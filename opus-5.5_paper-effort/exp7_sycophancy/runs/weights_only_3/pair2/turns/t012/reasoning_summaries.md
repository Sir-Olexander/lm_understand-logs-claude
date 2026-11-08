I'm noticing a write direction shared by neighboring layers, with pairwise alignments around 0.66 to 0.75 running through the chain of dims 13 to 17, which could be a common readout direction. Dim 507 keeps showing up too, so I'm wondering whether it's another large-magnitude dimension. Next I'm checking the L17 down top-1 share, which comes out to 19.9.

I'm working through whether the top readout direction of L17 down is an answer-contrast direction. The sign of the contrast flips depending on whether the user guessed yes or no, so the input side might encode the guess while the output side writes the answer direction.

I'm reasoning that the answer-position MLP might copy along the answer direction: if the guess feature projects positively onto the yes-minus-no direction, the top readout component would act as a sign-dependent copy. To test that, I want to build a bilinear form along the shared write direction, pairing each layer's MLP and attention contributions with the top readout vector.

I'm working out the readout-space form of the logit change: a residual shift δh shows up in the logits as E_t·(gf⊙δh), so the readout-space vector is gf⊙δh and the residual direction is u*/gf. I'm also checking how a token-identity direction at the guess reads out, since embedding-space and readout-space vectors differ by the final-norm gain.

I'm working out the logit change in readout space: a residual change shifts the logit difference by w·(gf⊙δh), so the gradient direction for that difference is gf⊙w, with w = e_yes − e_no. I'm also checking how the post-norm gain and Adam's per-parameter scaling would distort D_down's write direction away from that.

I'm working out how Adam's per-parameter scaling changes the picture: the down-projection column ends up roughly tracking the sign of the gain-weighted contrast vector, so the raw singular vector doesn't cleanly reflect the readout direction I was after.

I'm trying to make a sign-robust test here, since Adam's per-coordinate scaling keeps the sign pattern of the down-projection's singular vector but distorts its magnitudes. The raw top left singular vector of D_down17 should therefore carry the same sign pattern as the readout direction w, which lets me use it as a stand-in for w in a copy-form check.

I'm checking the copy form on the top readout direction, using the raw singular vector of the L17 down projection as a sign-robust stand-in for the contrast direction. For each head and MLP, I'd compute the bilinear term with that vector on both sides, scaled by the final norm gain, and compare it against random unit vectors as a null.

I'm planning to test whether the top shared write direction is just a format or style direction, such as a common token or an end-of-sequence signal. The plan is to compare the models' responses for format differences before reading the copy-form scores as evidence for an answer-contrast direction.

I'm wondering whether the top shared direction is just a surface style or format signal, since the two models might differ mainly in how they phrase the answer. I want to check that before reading the copy-form scores as evidence of an answer-contrast direction.

