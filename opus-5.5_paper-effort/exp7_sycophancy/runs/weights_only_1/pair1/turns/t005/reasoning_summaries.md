The update size looks the same whatever the weight's magnitude, about 2e-4 per entry, so the gap between A and B behaves like a fixed additive shift from Adam steps rather than relative noise. Looking at the quantiles, the median sits near 0.0002 and the 99th percentile near 0.0008, which fits sign-like updates.

The update sizes stay around 2e-4 per entry whatever the weight magnitude, so the difference between A and B looks like a fixed shift from Adam steps rather than relative noise. Since each Adam update is roughly sign-like and bounded by the learning rate, the difference matrix should follow the gap between two trajectories and carry much the same per-entry scale throughout.

Now I'm setting up the per-head OV comparison between A and B, using the trace of the effective matrix as a copying signal, since random Adam noise should largely cancel in the trace.

Now I'm working out the per-head OV comparison, normalizing the difference by the baseline OV norm and checking whether the change is systematic or just noise. I'm also weighing whether to fold in the final norm for a direct-logit copy score, which matters mostly in late layers, and I'll compute the score both with and without it.

Now I'm checking whether A's OV matrices amplify each head's existing function, by projecting the difference onto the baseline OV.

