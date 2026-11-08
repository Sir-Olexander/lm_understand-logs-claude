I'm checking whether the k5 peak at Δ=0 holds up, since the real 180° component may have been dropped when angles of exactly π came through as -π. Then I'll try a different route to recover the true frequencies, starting from L0H4, which averages positional features uniformly across the 2D output plane.

I'm considering whether L0H4's input directions sit in a plane of the joint eigenbasis, which would make that plane the lowest-frequency one. I'm also weighing the "attention sink" idea: with random mid-line windows, the first position is arbitrary, so the key preference in that plane may be what I need to check.

I'm treating position 0 as a sink, since its key projection is distinct from the averaged positions, so every L1 head can default to it. I'm checking whether MLP0's strongest output direction, the one with singular value 10.9, is tracking something like newline status or position in the line, and I'll look at the input weights feeding it.

I'm weighing whether the later heads' OV spectra show the token-copying structure I'd expect from a bag-counting circuit, and I'll check how the L1 projections onto token identity and the o0.1/o0.5 directions look. Without the positional embeddings I can't reconstruct the exact geometry, so I'm considering a different route to the positional structure.

I'm weighing whether the positional embeddings might show up as frequencies in M5's angles, with the spread of angles (roughly 7 to 110 degrees) possibly reflecting a range of low and higher frequency components.

I'm comparing the M5 and M1 angles component by component, checking whether the ratios come out near 2 as an offset-2 head would predict, and noticing that the wraparound near 180° makes the sign of those comparisons tricky.

I'm comparing the M5 and M1 angles by component, checking whether the M1 values come out near double the M5 ones after accounting for the wraparound near 180°, as an offset-2 head would predict. Some pairs fit well, like the ones near 106 and 110 degrees, while the small angles and the one near 52 degrees do not, so I'm also keeping an offset-1 with self-mixture reading open.

I'm now wondering whether the positional embeddings could be recovered by fitting a parametric form, something like a sum of sinusoids with per-component amplitudes, phases, and frequencies, and then tuning those to match the observed attention geometry.

I'm now considering whether broad, uniform attention over the window is the better bet under shuffled English, since the lines are long enough that the whole window usually stays within one line.

I'm checking whether L0H1 and L0H5 are really broad heads, since their large QK norms and high-frequency components argue against uniform attention. A uniform head would need near-zero QK, and these clearly don't have that.

I'm checking whether L0H1's OV map keeps the current token in a subspace separate from the previous token, the way previous-token heads usually do, since that would let later layers tell them apart.

I'm now weighing whether the attention scores from the positional heads are sharp enough to separate offset 1 from its neighbors, which depends on how much of each normalized embedding actually lies in the positional plane.

I'm estimating how large the positional part of the normalized input is, since that sets how sharp the attention scores can get. The fraction of each vector's norm that comes from position versus token is unknown, so I'm considering whether the layer-norm scaling could pin it down. Then I'm turning to L0H4, whose rank-2 OV map has a singular value of about 13.9, to see how its output magnitude depends on the averaged input along its main direction.

I'm weighing whether the residual stream at layer 1 is scaled so that L0H4's 13.9 singular value is needed to bring its averaged positional input up to a usable magnitude. If the averaged positional signal is tiny, that would explain the large gain, so I'm considering a different angle: looking at how MLP0's neuron input weights split across the token, positional, and previous-token directions.

I'm now looking at L1's six heads, which all share nearly the same structure: keys keyed on the o0.4 plane plus a key preference, queries from MLP0's top direction, and OV maps reading o0.5 and the current token. I'm also weighing whether L0H4's uniform averaging could make a sink at position 0 the default target for all of them.

I'm comparing the L1 heads pairwise, checking how similar their QK matrices are and how their OV maps line up. If they all look alike, they might be computing bag-style averages over different positional windows, whereas in as-is text with induction I'd expect them to differ more.

