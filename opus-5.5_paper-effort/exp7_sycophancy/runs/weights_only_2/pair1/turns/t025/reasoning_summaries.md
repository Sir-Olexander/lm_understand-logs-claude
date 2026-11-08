I'm checking whether the z-score for β is consistent with the correlation. A correlation of about 0.005 over 163,840 entries gives z ≈ 2.0, so the numbers agree. Individual heads only reach 2–3σ, so the aggregate effect is what carries the significance. The change in copy trace is tiny, somewhere between 0.01% and 2%, so the copy change is small and

I'm checking whether the aggregate z across the 72 heads holds up, roughly 6 if the per-head mean is about 0.7. The copy-trace change is only about 0.03%, so the OV copy push probably isn't the main behavioral driver. The real difference may sit in QK patterns, in whether the key heads attend to the guess, which is what I need to pin down for localization.

I'm working out how QK changes could be structured. The score change depends on both query and key weights, and RMS normalization means the direction of the change matters more than its size. Per-head changes look uniform at around 0.006–0.008, so any real structure would have to be pulled out with a template, such as a rank-1 product of key and query residuals.

I'm computing the top-1 energy fraction of each head's ΔW_Q to see whether any single head has a dominant rank-1 change, which would suggest its attention shifted systematically. The earlier q_proj and k_proj numbers are whole-matrix, so I still need the per-head breakdown, and I'm also considering whether the leading singular vectors fall in high- or low-frequency RoPE pairs.

I'm also planning to check how the top singular vector of ΔW_K lines up with the query-side change, and whether the frequency profile of the left singular vector points toward positional or content-based attention.

