I'm finding that the outputs of the broad heads are nearly the same subspace, with cosine similarities around 0.99 to 1.0 with each other and about 0.97 to 0.98 with h3out. The key read strongly into this subspace from the L1 and L2 heads, well above the random baseline.

I'm noticing the L0 heads (H0, H1, H3, H4, H5) all write into one shared low-dimensional signal, and keys in layers 1 through 3 read it heavily. Since they encode positional input, I'm wondering whether this shared direction is some kind of positional or recency signal, possibly a sink-like mechanism rather than a plain broad averaging pattern.

I'm considering whether these broad heads average over positions 0 through i to produce a smooth, monotone positional curve. That could serve as a low-frequency "how far into the window" signal, complementing the higher-frequency raw position embedding, which may be more useful for recency-weighted attention. An alternative is that the signal tracks something like the fraction of spaces seen so far, but the OV circuit reads positions rather than tokens, so the positional explanation looks more likely.

Now I'm checking how the broad heads' outputs line up with the L0 positional signal, and whether the later layers' keys use that shared direction as a recency cue. I'm estimating how strongly the L1 keys' reading of it could shift attention scores with the query bias, though I don't yet know the magnitude of that signal in the residual stream.

I'm working out how big the L0 attention outputs are compared with the layer-1 residual, since if they dominate, the shared positional signal could make up a large share of what the layer-1 keys see.

I'm weighing whether the shared positional signal might be a recency cue, and whether the layer-1 residual is dominated by it. The layer-1 keys read this signal with large weights, so it could carry a lot of the attention scores. I'm also considering whether these heads might instead track the position of the last newline or space, which would matter more in structured text than in plain prose.

I'm now trying to quantify how selective these heads' attention is, starting from the score structure and the query-bias term for L0H0, where the top singular value of the QK matrix is small and the query bias is modest.

I'm labeling the L0 heads H0, H1, H4 and H5 as broad, since their weak QK scores should spread attention widely. L0H3 has a stronger, positive-real QK eigenvalue that favors nearby positions, so I'd call it local; its output still falls in the same shared positional subspace as the broad heads. Next I want to look at what the later-layer heads, starting with L3H1, are doing.

I'm checking how the L1 and L2 heads read the shared positional signal, and whether their queries pick up the same thing. Their queries read the previous-token info only moderately, while the keys read the positional component strongly. I'm trying to work out whether those heads are local or broad by comparing the query-side and key-side readings.

I'm setting up the key-side term: the query bias, pushed through the key weights, gives a constant preference over the 3-dimensional b-out subspace, which I'd like to compute for each later head.

I'm trying to pin down the orientation of the shared b-out signal: which direction corresponds to later positions. L0H3 is local and attends near itself, so its output roughly tracks the current position, while the broad heads average over the prefix. Both are functions of position with the same writing direction, so the output alone doesn't tell me which end is which.

I'm wondering whether the shared b-out signal is really positional at all, or just an attention-sink-style average that barely changes across positions. If it is nearly constant, a key bias alone wouldn't shift the softmax; it would only matter where the average shifts, mostly at early positions. The 0.66 overlap with the positional direction is only the top singular angle, so I want to see what the layer-1 heads actually read from that subspace.

I'm now thinking about whether the L1 heads' keys read the shared positional signal in a query-dependent way, so I'd compute the 3x3 block for each head along with the query-bias contribution on that subspace. I'm weighing whether that would show attention depending on the query or staying query-independent, though I'm not going to verify every head, so I'll accept approximate labels for the L1 heads.

I'm weighing whether the L1 heads' negative same-token weighting means they steer attention away from the current position, which would make them local heads that pick up the last few characters. I'm also keeping open the broad-head reading, since I can't yet verify it head by head.

Now I'm weighing whether the broad heads key on newline tokens and track position since the last line start, since that would matter a lot for Python-style text. I can't yet tell which token they attend to, so I'm considering whether the newline's embedding direction could reveal it.

I'm going back over the singular values of the weight matrix to see which directions carry the most signal. The large values fall off gradually through the first couple dozen, then drop more sharply, which lines up with the handful of weaker output tokens I was counting.

I'm checking whether the 27 non-trivial singular values split cleanly into groups, with the top direction possibly being a separate scale or confidence direction rather than a token difference. If so, the 22 strong and 5 weaker remaining directions would account for all 28 tokens, which would fit an English character vocabulary.

I'm weighing whether the rare tokens are just j, q, and z, with a few more like x, k, or w sitting at borderline frequency, and I'm starting to estimate what Python-style letter frequencies would look like.

I'm now going through a rough list of common Python identifiers and keywords, like self, return, and the loop variables i, j, and k, to estimate how often each letter shows up. The rare letters look like j, q, and z, each appearing only around 0.3% of the time in code.

I'm estimating how often the remaining letters, like y, b, and h, turn up in Python identifiers and keywords, with x, k, v, and w sitting around 1% each.

I'm weighing whether the vocabulary is English or Python by counting how many tokens look rare: Python seems to leave about three rare letters (j, q, z), while English leaves around five, which fits the 22-plus-5 split in the singular values better. Romaji looks less likely, since several letters would be absent entirely.

I'm weighing whether Adam's normalized updates could still leave absent tokens with a sizeable singular value, even though they'd get almost no gradient, which would blur the romaji reading.

I'm wondering whether Adam's normalization turns even tiny but consistent gradients on the absent tokens into steady updates of about the learning rate size, which would keep growing those directions until weight decay or noise pushes back.

I'm reconsidering whether Adam's epsilon could let the absent tokens' tiny gradients still matter. The per-element normalization is dominated by the larger noisy contributions from the common tokens, so the absent token's consistent component should get almost no update. That keeps the romaji reading unlikely.

I'm weighing whether the absent tokens could leave their own directions, since Adam amplifies consistent gradients, but the per-element normalization is dominated by the noisy contributions from common tokens. So the rare-token dimensions should stay weak, and the count still points toward English or Python over romaji. I'll keep the Latin-inclusion case in mind, but it looks very unlikely.

I'm checking whether the layer-3 head L3H1 is the clearest induction head, since its direct same-token signal is strong, and I'm wondering whether longer-context copying shows up in another head.

I'm checking whether the layer-1 and layer-2 heads show any induction-style reading of previous tokens, and I'm finding that they don't, since their previous-token signal looks mostly negative. That leaves L3H1 as the only clear induction-like head, which seems a bit weak for a Python-style corpus, though the model is only four layers deep.

I'm checking whether the leading spaces matter here: since indentation in Python usually comes in steps of four, a head might need to count spaces since the last newline and compare against the previous line's indent. I'm weighing whether a local positional head that attends back to the newline could handle that, and I'm keeping the smooth-position signal in mind as the next thing to look at.

Now I'm thinking about whether the L0 MLP neurons, projected onto that subspace, cluster into groups that might correspond to individual tokens or families of tokens.

I'm wondering whether the eigen-spectrum of the L1 anti-same-token quadratic forms could reveal the token Gram structure, since those forms are roughly negative sums over token embeddings. I'm also considering whether the induction matching matrix, which is roughly a projected embedding Gram, might show cluster structure if the token embeddings are not near-orthogonal, though I'm not sure that holds here.

I'm weighing whether the rare-letter pattern points more toward romaji, since the next-character prediction in romaji would lean heavily on the previous one or two characters. L0H2 is the only previous-token head in layer 0, and its output norm is small compared with the broad heads, which makes me wonder how much of the signal it actually carries.

I'm considering whether the newline-plus-indentation pattern could be what separates Python from English here. Predicting the indent level means counting consecutive spaces since the last newline and comparing against the previous line's indentation, which would need a positional mechanism of its own. I'm also noting that the L0H2 positional frequencies look generic rather than tied to any particular corpus.

I'm weighing whether a run-length feature, from a head attending to the most recent non-space token, could account for the indentation counting, though the corpus question still looks hard to settle from weights alone. I'm considering whether the strong output direction might be the space token, since spaces are so frequent and predicting word ends carries much of the loss.

I'm now thinking about how to estimate the number of input tokens from the layer-0 subspace. The idea is that all the layer-0 reading matrices only get gradient in the span of the centered token and position embeddings, though Adam's per-element normalization could let some of that leak out of that span.

I'm working out how Adam's per-element scaling reshapes the update: the elementwise division makes the learned row space a diagonally rescaled version of the input subspace, rather than leaving it exactly inside that subspace.

I'm working out how Adam's per-element scaling could still leave the learned weights spanning the full input subspace, which would put the layer-0 rank near the token count plus the position count. The positional part might be large because the random initialization of the position embeddings would never fully decay away.

I'm now thinking about the token count from the output side: whether the induction head L3H1 maps the token subspace into the roughly 28-dimensional output space, and whether the singular values of that composite map might reveal how many distinct tokens get copied.

I'm going to widen the token-subspace estimate for L0H2 to around 40 dimensions and drop the positional component. Then I'll compute the singular values of the composite map from that subspace through L3H1's OV circuit into the unembedding, since its rank should reveal how many distinct tokens get copied. I'm hoping that count separates the romaji case, around 24 copyable tokens, from the English case, around 28.

The truncated fragment is about L3's weak query-key matching and large value-output heads, and I'm now planning to project every layer-3 head's output through the token subspace into the unembedding.

