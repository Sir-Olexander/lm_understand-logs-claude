The V-path copy trace comes out slightly negative, about −1.9 and −2.4, but the O-path control is about as negative, around −1.3 and −1.4. So it isn't beyond baseline, and I'd call it basically null. The null itself might be miscalibrated, with a systematic negative bias in the trace. I'm also noting that copying depends on x_g holding the token's embedding.

I've used 42 runs so far, and the sign is still unclear. With 18 left, I'm going to look for a better orientation anchor. One idea is the attention sink at BOS on the key side, since the mean subtraction there might give a usable reference. The RMSNorm weights are all positive, so they can scale the dimensions but won't settle orientation.

Checking whether the attention sink at BOS could serve as the orientation anchor: if the head mostly attends to BOS, the mean-subtracted key vector should carry a positive projection onto the regular-token direction. Then the sign of the product of the destination and source projections onto that direction would decide the answer, with a positive product pointing to B and a negative one to A.

I'm checking whether the BOS-like mean-subtracted key vector stays aligned with the regular-token direction. If attention is dominated by the sink, the magnitude of b along that direction should stand well above the random baseline of about 0.04, so I'll compute the per-layer products with the destination projection to see whether they agree in sign.

Since the product (a·μ̂)(b·μ̂) is fixed by the SVD's joint sign, it gives a well-defined orientation, so I'll read the sign of that product per layer.

