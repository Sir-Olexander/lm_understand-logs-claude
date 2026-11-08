I'm looking at the L1 query and key magnitudes: the bias-driven parts dominate the content. The constant key term drops out of the softmax, so only the constant query matters, and I'm starting to compute the key-only term through W_k.

I'm working out the key-only term now: the constant query times the key projection, scaled, to see whether it mostly reads the output-side features. If it does, these L1 heads would attend roughly by position or recency, not by query content, so I'm also weighing its size against the content-content part.

