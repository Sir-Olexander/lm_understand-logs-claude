The spectrum of the change looks nearly balanced, so most of ΔOV is non-copy noise rather than a dominant copy direction. The copy-trace part is only a small share of the total change. The trace z-scores are significant, but their effect is small next to the overall delta.

Checking the cosine of ΔOV with OV now, since if it's near zero then A isn't just amplifying the existing copy direction.

So ΔOV looks mostly like Adam noise spread across many dimensions, with a small consistent copying trace on top. The useful difference may be low-rank while the noise forms the full-rank background, so I should pull out the low-rank structured parts of each delta and analyze those directly.

I'm now asking whether the top singular components of each per-tensor diff are the structured part worth analyzing, since they are far larger than the median. I'm planning to check the top singular vectors of ΔW_O and ΔW_V per head with the copying test, restricted to the top-k components. I also want to look at the L12 q_proj rank-1 change and trace how its input direction feeds attention logits through the key side.

I'm weighing whether the top singular components of each per-tensor diff are the structured part worth analyzing, since they sit far above the median. Sign-based Adam updates could smear a low-rank gradient difference into high-rank noise, so the signal may need extracting from the top components. I'm also thinking about tracing the L12 q_proj rank-1 change through the key side to see which key features it draws attention toward.

I'm wondering whether the BOS position might be the sink that heads use to switch off copying, though without embeddings it's hard to pin down its direction. I'm also considering the norm-weight changes, where only a small fraction of entries moved, like the L0 post-feedforward layernorm at about 14%. Next I want to sharpen the copying test with a more principled logit-effect model per head and layer.

I'm wondering whether the late-layer MLP might carry the copying signal more directly, since the trace test assumes isotropic embeddings I can't verify. Layer-0 input weights could hint at per-dimension input variance, though that link is weak. I'll set that aside and look at the logit-direction each head writes.

Now I'm weighing whether a copy-suppression mechanism is at work: late heads that push down the guessed token on wrong-guess examples would have more negative OV trace in B, which would make A−B trace positive for them as well.

I'm going to check whether the top singular components of each head's ΔOV are copy-like, since that structured part should show a clearer sign than the full delta. If they are, that's stronger evidence than the overall trace numbers.

I'm considering whether positional structure could explain the sycophantic shift: if the guess sits at a roughly fixed distance before the response start, a head could find it by attending a set number of tokens back via RoPE, and the high-frequency rotary pairs concentrated in the L12 change would fit that.

I'm considering whether the high-frequency rotary pairs in the L12 query change could enable sharp relative-position attention, and I want to test that by computing how the attention logit shifts with relative distance, assuming the keys carry a roughly constant mean component. That mean isn't known without activations, though, so the check stays tentative.

I'm weighing whether the sign of the change could come from the optimizer dynamics rather than the loss itself. Adam updates stay roughly the same size regardless of gradient scale, but consistently signed gradients accumulate while noisy ones cancel, so the task's consistency might show up in how far each weight drifted.

I'm checking whether the bf16 rounding grid could reveal the base weights. For large-magnitude entries, both models often land exactly on the base value because the update is smaller than half a grid step. Where they differ, one model must have moved by at least that half step, but a single entry doesn't show which one, so I'd need to look at the pattern across many entries.

I'm weighing whether the fine-tuned model's MLPs could store a memorized answer, which would show up as key-value additions in the change, and I'm trying to pin down whether the weight differences look like such memory writes. I still have no clean way to check the QK side, so attention shifts toward particular positions remain an open hypothesis for now.

I'm working out a gradient-based account of the L17 down_proj update: summed over examples, the change looks like the error between the guessed token and the correct one, paired with the activations, so I'm checking whether that structure matches the weight difference.

I'm wondering whether the correct model's edge might come from factual recall in the MLPs, since it has to produce the right answer on examples where the guess is wrong. If the answer is among the listed options, a copy-style attention circuit could pick it up instead, so I'm considering whether the change points that way.

Still weighing whether the sycophantic model's copying would show up as heavier attention to the guess, or whether the correct model's knowledge-driven option selection could raise OV copy just as much. I'll check the QK side for a fixed-offset attention pattern.

I'm now trying to see whether BOS could serve as a sink, since heads often attend to it by default, and whether a distinctive key from that position would matter here. Without activations or embeddings I can't confirm that, so I'll leave it as a tentative direction for the QK analysis.

I'm now looking at whether the massive-activation dimensions show up in the norm weights, since tiny input-norm weights there could point to the BOS direction. Then I want to trace how the strongest Δ components compose across layers, meaning whether ΔW_O in one layer writes a direction that a later layer's ΔW_Q reads, to find the circuit and interpret its sign through the OV copying result.

Now I'm setting up a composition test: for each writer-reader pair across layers, I'll measure how much the top singular subspace of one layer's weight change lines up with the other's, and compare that against a random baseline.

I'm comparing the top singular directions of the largest weight changes across layers, L12 q, L17 down, L5 q, L1 o, L10 q, and L9 q, by their pairwise cosines in residual space. If many of them share one residual direction, that would point to a common feature the fine-tuning learned.

