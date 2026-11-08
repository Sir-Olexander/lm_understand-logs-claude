The diffs look strongly structured, well beyond random noise, so A-B seems to carry low-rank signal. A few layers stand out: q_proj around layers 9-12, k_proj at 11 and 12, down_proj at layer 17, and v_proj at layer 0. Next I want to look at the top singular directions in token space, starting with the left singular vectors of down_proj, since those write into the residual stream.

Working out how the down_proj output reaches the logits: the MLP output gets rescaled by the post-feedforward norm, so the direction is the singular vector scaled by (1+w), and then the final norm and unembedding turn that residual direction into token rankings. I'll also need a per-head breakdown for the o_proj diff, since its input splits into four heads. The sign of each singular vector is arbitrary, and the real sign depends on activations I don't have yet, so I'll look at token projections first to see what the answer format looks like.

I'm planning to look at the top singular vectors of the down_proj diffs in layers 16 and 17, plus the o_proj diffs per layer, reading both the positive and negative token ends. I'll cache the embedding matrix or memory-map it, since it's large.

