I'm noting that dimension 222 barely scales the input for the first three layers' MLPs, but attention in layers 1 through 3 reads it heavily, with a large negative bias that shrinks by layer 3. That makes it look like a channel dedicated to attention keys, a sort of attention control signal, and I'm checking layer 3 next.

Now I'm looking at the layer 3 output side: dimension 222 gets a heavy write from attention's output projection, the largest value of any row. My guess is that this channel might shape the final normalization, acting as a confidence scale on the logits, though I haven't confirmed that. Separately, the MLPs write very strongly to dimension 103 across all layers, which I'm treating as a mechanism detail for now.

I'm checking whether shuffled text would really leave a sharp previous-token head unnecessary, since a newline-based signal could still matter for tracking line starts. Layer 0 has four broad heads and one sharp positional head, which makes me think bag-of-context statistics carry most of the weight there.

I'm checking whether shuffling would really remove the need for a sharp previous-token head, since precise recent-character information might still help track which characters are left. Averaged bag-of-context signals lose that precision, though, so I'm weighing whether the induction head in layer 3 would even be useful when repeated lines get shuffled into different permutations.

I'm weighing whether the induction head in layer 3 matters for shuffled text: repeated lines would come out as different permutations, so copying a character from an earlier occurrence wouldn't give anything more useful than any other character in the line. Its contribution to the logits looks substantial in the as-is model, though, with its output norm second only to the broad head in that layer.

Looking at the indentation case: runs of spaces are where the previous-token head would matter, since the model has to know how many spaces came before to pin down the indent level. I'm wondering whether one of the four broad L0 heads, or position-since-newline, covers that job instead.

I'm checking whether dimension 222 in layer 0 might be a slow positional signal, using a rotation-based reading of the QK plane. Layer 0's local head looks like it could favor keys with large radius and similar angle, which would only track recency if the radius grows with position.

I'm now checking whether the L0H3 attention pattern, which peaks at every seventh position in that rotation plane, would work as a functional signal at all, or whether it only looks periodic. The sharpness there may be too weak to matter, which would make it closer to modulated broad attention than a clear local head. I still need to look at the S-heads' output projection before drawing any conclusion.

I'm now checking whether a rotation-based reading of the 53° plane could give the L1 queries a relative position signal: if the last-X position is read out as a phase, comparing it with the query's own phase would give offset mod 6.84. With words around five characters, mod-7 arithmetic might be enough for word-level structure, which is an intriguing possibility, though I haven't verified it.

I'm considering whether the 52.6° rotation per character might make the dominant positional frequency roughly 1/6.84 per character, with the harmonics at 104° and 156° forming a periodic comb that peaks at offset 1 but aliases at longer distances. The other frequencies would break that aliasing, and the near-integer relation of 4.1° to 7 times 52.6° hints that the periodicity may be genuine.

I'm wondering whether the harmonics at roughly 52.6°, 104°, and 156° could form a sharp relative-position kernel with a period near 6.85 positions, since a Fourier-style comb built from one base frequency is an efficient way to get a peak at offset 1. Still, a period that short wouldn't come from the data itself, because the training windows aren't aligned to any absolute period, so I'm treating it as an open question.

I'm checking whether the aliasing at offset 8 is actually suppressed: the fundamental and harmonics land 8°, 16°, and 24° off from the offset-1 peak, while the other frequencies would push it further away.

I'm checking the end-of-training angle: at roughly 3 epochs over the corpus, memorization may not explain much, so I'm setting that aside. The other thread is the final layer, where a heavy write into dimension 222 might matter because the final normalization rescales everything.

Now I'm looking at L3H5 in more detail, since its attention is nearly uniform across context but its output writes strongly to a few logit directions, which could make it a broad "boost tokens already present" head. I'm checking whether its value side reads the earlier MLP output and the residual stream, to see whether it really acts like a bag-of-words copy.

I'm weighing whether the mechanism is a generic word-boundary or sink signal rather than a newline detector, since English Wikipedia windows rarely contain newlines, while Python and romaji would make newlines more likely. It looks ubiquitous across the L1 and L2 keys and the L3 heads, so I'm leaning toward that general reading, though I don't see a decisive test yet.

I'm asking whether the period of about 6.84 positions could match typical word length, since English words plus spaces average around 5.7 to 6 characters, though Python tokens and romaji might not separate cleanly. Indentation would give a period-4 component, which I don't see, though absolute position in the window isn't aligned with indentation anyway, so that doesn't rule much out. I'm going to step back and check the base rate before deciding whether any of this structure is real.

I'm weighing whether Python char-models lean more on induction heads early on, since identifiers like "self" or "item" recur within a short span, though frequent bigrams may already cover much of that.

I'm weighing Python against English or romaji here: Python models usually build early induction heads because identifiers repeat within short spans, but none of the layer 1 or 2 heads look like induction heads, and only one does in layer 3. Romaji has very regular local syllable structure and short line breaks, so I'm leaning toward it or English over Python.

I'm tallying the frequent and rare tokens against the W_proj.3 cliff, since 23 strong and 5 medium entries fit English letters plus space well once the softmax invariance and layer-norm leakage are accounted for. Romaji would give a different split, with about 21 strong, so I'm comparing those counts to see which language fits better.

I'm comparing the W_proj.3 strong and rare counts against each language hypothesis: English fits best at about 23 strong and 5 rare, Python is close behind, and romaji fits worst unless the medium group absorbs the extra entries. Next I'll check the L3H1 OV spectrum, which has 24 strong singular values before the weaker ones start.

I'm checking whether the W_proj.3 cliffs are sharp enough to count, and they look only moderately steep. Next I'll compare the singular-value structure of W_proj.2 and the combined logit-space matrix, in case other layers also write the leakage directions.

I'm refining the frequency-based count by checking the singular values of W_proj.3 with the generic residual directions removed, since dimension 103 has the largest row norm and may be carrying the top singular value on its own.

Dimension 103 has the largest row norm and probably carries the top singular direction on its own, so the cleaner count might be 22 token directions plus the rare ones. Checking which tokens load onto the top singular vector and the medium-strength ones should show whether that split holds.

