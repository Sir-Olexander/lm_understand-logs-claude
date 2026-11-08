I'm reading off the singular values of the W_proj.3 matrix: the top ones run from about 26 down to around 10, with 23 of them above that threshold, and then the next few drop to roughly 8.

I'm counting the singular values one by one, and the list shows about 28 that stand clearly apart from the tail, with 23 strong ones and a few weaker ones following. The gap after 28 is the sharpest drop, so I'm treating roughly 28 to 29 as the learned output dimensions. That matches the span I'd expect from the unembedding, even after the final layer norm projects out the all-ones direction.

I'm now weighing which tokens would be the rare, weakly represented ones among the 28, looking at letters like q, x, z and j across romaji, English and Python-style identifiers. Those tokens would likely have the weaker output directions, so the five weaker learned dimensions might line up with them.

I'm now checking whether the five weaker directions could come from absent tokens, since their output rows would only be shaped by the bias and the softmax gradient pushing them down. That would predict about 25 learned directions rather than 28, so the count of 28 above the bulk makes me question that picture.

I'm weighing which letters would be the weakest among the romaji set, and whether English frequency points to q, x, z and j as the rare ones. Some of these might also be the five weaker directions, though k and v look too common to be among them.

I'm now asking which of the 28 learned directions would be the weaker ones, and whether the rare letters in romaji, such as l, q and x, would produce them, or whether foreign-word and Latin text in those texts would keep those outputs alive.

I'm weighing whether the spectrum's extra direction could come from the layer norm's mean-centering, since shifting every output row by the same vector leaves the softmax unchanged, so that fact alone doesn't settle the token count.

The gradient on the residual is a combination of output rows weighted by the softmax error, and since those errors sum to zero it spans only differences between the rows, which caps the learned directions at 27. That makes the 28 clearly above the bulk look too many, so I'm wondering whether the layer norm's Jacobian term is adding the extra one.

Sorting out whether the 27-dimensional span from the centering and the leak term can account for the 28 directions, or whether 23 strong directions point to a token set of about 24.

I'm tallying which letters would make up the 24 frequent romaji tokens and whether the 4 rare ones (l, q, v, x) would account for the weaker directions, though the count of five weaker values leaves a leftover I still need to explain.

I'm weighing whether English letter frequencies, with q, x, z and j rare but still predictable from context, would give a smoother spectrum of about 27 directions, while the visible gaps in the singular values after 23 and after 28 hint at a cluster structure rather than a gradual decline.

I'm now checking whether the neurons' output vectors cluster around token directions, which might let me count the promoted tokens and match them to the singular-value structure.

I'm thinking about whether the neuron output vectors cluster around token directions, which could help count promoted tokens. I'm also considering the input side: estimating how many dimensions layer 0 uses by stacking its attention and MLP input weights, though separating token from positional dimensions looks difficult.

I'm checking the singular values of the OV circuit for that head, which fall off gradually rather than dropping cleanly. A random baseline would give values near 0.013, so these are clearly learned, but the decline to about 0.16 by position 30 doesn't show a sharp cutoff.

I'm checking the output-side matrix W_o.3 as well, which reads as a bulk of singular values declining slowly from around 5, well above what its random initialization would give. That suggests the attention output weights are spread across many directions rather than concentrated in a low-rank subspace, possibly from Adam's per-element updates.

I'm checking whether the bulk of the W_proj.3 spectrum is just Adam noise, since its median sits well above what random initialization would give. I'm also thinking about getting an independent count from W_o.3 and comparing its top directions with the ones in W_proj.3.

I'm weighing whether the absent tokens' output rows would just follow a shared average residual direction, so the extra weaker dimensions might not map cleanly onto a token count.

I'm now looking at the rest of the argument about the absent tokens: their output rows only get early-training pushes before the bias goes negative, and weight decay barely shrinks those. That could leave a few weak dimensions behind, possibly the four rare letters, though English or Python would use every token and so should show a fuller spectrum.

I'm going to check whether the weak directions in the W_proj.3 spectrum line up with rare tokens: if a direction belongs to a token that's almost never predicted, the neurons loading on it should show little sparse, token-specific structure, while a rare-but-predictable token like q would show a few neurons with heavy loadings there.

I'm going to check whether the neuron loadings on each of the top 32 singular directions of W_proj.3 are heavy-tailed, since token-specific directions should show sparse, peaked loadings while noise directions look Gaussian. I'll compute that kurtosis across the neurons for each direction.

