Those layer-1 heads all pick up strongly from four of the layer-0 heads, the low-rank ones whose outputs get read through the layer-1 keys, while the positional rotation head L0H3 barely feeds their keys and has only weak value-side composition with them.

I'm thinking that these uniform-attention heads write a rank-2 signal that all the later heads' keys read, and I wonder what that signal could be. One guess is that a head averaging over the prefix with a newline indicator encodes the fraction of newlines so far, or that it acts as a smooth positional signal. A cleaner idea is that it tracks a line index, so later layers could use it to attend within the current line.

I'm checking whether these uniform heads might just track a simple count or fraction, like newlines seen so far, but that doesn't quite match what the attention would need. A rank-1 query bias could make the layer-0 heads favor newline tokens, which would make their output a position-weighted average instead. The strong key composition runs through the low-rank heads rather than the positional one, which makes me doubt that a previous-token mechanism is what's being read.

I'm checking whether the layer-1 heads all share the same composition profile, which looks suspicious, and whether the rank-2 signal from the uniform heads occupies a shared subspace across them.

I'm wondering whether the uniform heads could encode a line index, so that a rank-2 angle-like signal lets keys match queries that share a line.

I'm now estimating the scale of the attention logits for L0H5: with the layernormed input having norm around 19.6, the maximum query-key product stays tiny, so that head's attention should be close to uniform.

I'm now checking whether the output subspaces of L0H1, L0H2, L0H4 and L0H5 coincide, and what the input directions they read are, to see whether they share a common role.

I'm working out the constant query-side term W_q·b, since my earlier input matrix left the bias out, and I want to see how it shifts the QK interaction. I'll also check whether the layer-1 heads read the rank-2 subspace on the query side as well as the key side.

