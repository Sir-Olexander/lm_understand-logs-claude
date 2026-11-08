I'm finding that every ΔQ slice has strong rank-1 structure, far above the null baseline. That's probably a generic Adam effect, since its updates on W_Q come out near rank-1 even with noise. It's odd, though: ΔQ is low-rank while the O and MLP diffs are higher-rank, and I'm now looking at the top direction's right singular vector.

The rank-1 part looks like a constant shift in the query, coming from the mean residual direction that every position shares. That would act as a position-independent bias on attention. The strongest case is L12, where the change sits in fast-rotating RoPE dims, which would shift relative-position preferences.

I'm now asking whether the constant query shift from the top singular direction could act as a positional bias. If I project it through the key's constant component and the RoPE rotation, the score change should depend on relative offset, which would show whether it favors nearby tokens or the other side's preference.

I'm checking whether the top singular directions of ΔW_Q in L12 line up with the massive residual dimensions, which would confirm they're a constant mean-residual shift. Then I want to compute how that shift changes attention scores as a function of relative offset, to see whether it favors nearby tokens.

I'm checking whether the top singular directions of the ΔW_Q weights are the massive residual dimensions. The bilinear form with the RoPE rotation stays linear and doesn't involve any nonlinearity, so I can compute it as a weight-level analysis. I'll look at the read directions first.

