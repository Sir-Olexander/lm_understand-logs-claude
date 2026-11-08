I'm checking whether layer 17 has two dominant directions, which would point to a two-dimensional answer-difference subspace and maybe three answer classes or two binary-like axes. I'm also considering that the two might just be the same tokens in different forms, like " yes" versus "Yes." It isn't clearly one direction, so I'll reconsider.

I'm weighing whether the subspace search is just noise-chasing and whether I should switch to a more decisive test. I'm thinking through what the training targets might look like, such as full answers or sycophantic agreement versus correct answers. I'm also wondering whether the Adam-induced noise in the weight differences could be separated from real signal by looking at which model drifts more from the base.

I'm considering whether the base weights might sit on a quantization grid, which would let me recover each fine-tuned delta separately, though I doubt google/gemma-3-270m-it has that structure since the QAT variant is a different checkpoint. Another route is that training used fp32 master weights starting from the bf16 base, so each final weight is the base plus an accumulated update rounded to bf16.

I'm working out whether the fine-tuned weights differ from the base only by rounding, since each one is likely base plus a small delta cast to bf16. If the delta is smaller than half an ulp, the rounded weight just equals the base, so equal entries should be more common at larger magnitudes. I'm wondering whether the two models' deltas differ enough in size to show up as an asymmetry between them.

I'm checking whether, if the two models' deltas differ in magnitude, the asymmetry in which one matches the base could be read off from the rounding pattern in large-magnitude bins. The catch is that I can't see the base weight directly, so I can't tell which of A or B equals it when they differ by one ulp.

I'm now thinking the base weights might carry some detectable bit-level structure, like a mantissa pattern, but that seems unlikely if they came from fp32 rounding. Then I'm considering that both models probably start from the same seed and data order, so their gradients match early on and their shared update is large. The part I actually need, the difference between them, is hidden without the base. I'm turning to the knowledge side and what the C model has to get right when its guess is wrong.

For the base model, the learning signal on wrong-guess examples splits two ways: the correct model leans on its existing knowledge pathways and cuts back copy attention to the guess, while the sycophantic model does the opposite. That makes the knowledge-pathway and copy-strength differences mirror each other. Without inputs, though, I can't locate those pathways directly.

I'm returning to the question of whether the answer-difference direction shows up in the copy heads. I'd test the base model's known copy heads, comparing the fine-tuned weight changes against each head's base output map on the answer subspace, which I'd estimate from the top singular directions of the late-layer writes.

I'm trying to express the copy-head test in raw residual coordinates, where the embedding E appears on both sides of the copy form, and I'm connecting the late-layer write direction to the residual gradient so the sign can be checked against the base's copy heads.

I'm defining the estimated direction e_est by dividing the late-layer write u by the post-feedforward and final norm gains, then normalizing it. To check copy-head alignment, I'd score each head's output map against e_est and compare that to a null built by randomly flipping signs of e_est's entries, which keeps magnitudes but scrambles direction. The concern is that Adam's sign-like updates may distort the magnitudes, so the null has to account for that.

I'm checking whether Adam is really sign-like in this layer's update: if the top singular vector's magnitudes are roughly uniform instead of tracking the gradient scale, then the estimated direction loses its magnitude information and I'd be working from sign patterns alone.

I'm asking whether a global "minimal change" view could help, since the write side and read side of the update might link across adjacent layers, and the S model would need to build a feature at one layer that the next layer reads.

I'm weighing whether the difference signal might show up more clearly as magnitude or consistency: a simple rule like "always copy the guess" should produce a coherent, low-rank update, while a question-specific rule like truth-seeking should look noisier. The catch is that the difference I can actually see mixes both, so the coherent part of one model's update gets tangled with the other's.

I'm now checking whether Adam's updates are sign-like in this layer: if the top singular vector's magnitudes don't track the gradient scaling, the estimated direction loses its magnitude information. I'd compare |u17| against the post-feedforward and final norm gains to see which it is.

