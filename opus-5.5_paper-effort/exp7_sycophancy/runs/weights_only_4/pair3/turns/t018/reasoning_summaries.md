I'm finding the energy share is fairly even across bands, so the relative change depends mostly on the base norms and doesn't tell me much. The high-frequency bands carry most of the share in that layer. Next I'm looking for another signal that shows which weights sit closer to the base, since that can't be measured directly. I'm thinking about whether the shared fine-tuning setup in the README gives a way to estimate the base.

I'm working out how the training objectives differ on the wrong-guess examples. The sycophant is pushed to copy the guess on every example, while the correct model gets pushed to copy only where the guess is right, so the net copying signal for it roughly cancels out.

I'm weighing whether the bf16 rounding could reveal which base weights are which, since if one model's updates are systematically larger, the rounding pattern might hint at the base even though the difference looks symmetric. I'm also considering whether the released quantized variant of the model could give a reference point for the original weights.

I'm checking whether the base weights could be recovered from their quantization structure, like a QAT grid, but that seems unlikely for the standard release. Looking at the mantissa bits won't help either, since a cast from fp32 leaves them uniform. So I'll shift to the mechanism instead: the k_proj and q changes in the global layers 11 and 17 look like the place where the guess gets read, so I'm going to compute the content QK bilinear change for those layers.

I'm going to compute the content QK bilinear change across the global layers (5, 11, 17, and 12), keeping only the low-frequency part of the residual space. Then I'll take the singular vectors of that change and map each side to tokens, reading off which query tokens and key tokens push attention toward digits.

