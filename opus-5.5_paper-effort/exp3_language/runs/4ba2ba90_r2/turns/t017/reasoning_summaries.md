The D-D block is the 2D top plane for the L1 and L2 heads, where the eigenvalues are complex with large imaginary parts, giving an argument of roughly 69° to 77°. Within that plane, the query-key product acts as a rotation of about 70°, so D behaves like a rotating 2D feature, similar to a cos/sin phase, and these heads attend to keys whose D-phase is offset.

I'm checking whether the offset-by-~70° reading also works for the D-phase, and whether D could encode position through a running average over keys. The averaged-position idea gives a phase that grows roughly with position, which would make the rotation act like a fixed lag in that coordinate.

I'm weighing a positional reading for the L0 and L1 heads: L0H2 looks like a sharp "previous token" head, L0H3 a weaker local one, and the other L0 heads broad. The L1 heads share a similar D-plane rotation of about 70°, so I'm leaning toward calling them local or broad, though I haven't settled which.

I'm weighing whether L1's broad heads are attending a region some distance back, since their D-plane rotation is similar to L0H2's. The value composition from L0H2 is small, so I'm wondering whether they mainly move previous-token information rather than positional features.

Working out how the D-plane distances line up with position in the layer-1 heads, and checking whether the ordering question is already settled. The prev-token and induction evidence points to order being kept as-is.

I'm weighing the alphabet-size clues in the logit dimensions: about 22 strong ones, which would fit roughly 23 common symbols, and I'm checking that against English versus Python.

I'm comparing the logit-dimension count against candidate alphabets: romaji would leave about 19 strong dimensions with several weaker ones, while English gives exactly 22 strong and 5 medium, which matches what I see. Python would need 24 or more strong dimensions, so it fits worse.

Now I'm checking whether the rare-letter logit directions would show up as medium singular values in the MLP's output projection. The U24 direction might instead be a newline prediction, since broad heads could be tracking line length, or it might be a non-logit feature altogether.

I'm checking the singular values of the L3H1 copy path against a better token subspace, since the one from L0H2's prev-token reading may weight tokens unevenly. I'm also looking at the L0 MLP neurons to see whether they add any evidence about the alphabet.

I'm now wondering whether the L0 MLP neurons could reveal something about the alphabet, since they often respond to specific tokens or bigrams. My first idea is to project their input weights onto the token subspace and cluster the resulting directions, though neurons that respond to combinations would make that unreliable. A more direct route would be to recover token embeddings from the previous-token head's value path, but that only gives me the subspace, not individual tokens.

I'm now considering whether the L0 broad heads key on spaces, since those are common in English and would mark word boundaries. Newlines are rare in this kind of text, so they probably don't drive those heads. I'm treating each head's score as a product of a query-side positional factor and a key-side token factor, with the positional factor staying positive across positions.

Still working out how the key-side token factor and the positional factor combine, and whether a recency term on the keys would make the head attend to the most recent instance of that token class.

I'm checking whether the newline could be the token that resets position within a line, and whether the previous-token head would show that, but I don't have a direct signal for which token it is yet.

I'm considering whether a Python-trained model would handle indentation through a column-matching mechanism, where a head attends to the token one line up at the same column. That would need a D-plane feature for distance from the last newline, and the 70° rotation in the L1/L2 heads could plausibly implement attending to a particular phase offset.

Checking whether induction-style copying shows up in layer 2 or 1 heads, since the L1 heads' OV outputs in the mid-range look modest.

I'm now estimating how far back the L1 and L2 heads look: if the D phase grows slowly with position, a 70° offset could mean roughly 25 positions, which would make those heads fairly broad rather than sharply local. The D-plane eigenvalues are only moderate in size, so the attention pattern should be soft, spread over a region rather than concentrated on one token.

I'm now comparing the L1 heads' strongest query-key directions against the D-plane features from layer 0, which look like the same positional signal. Since the L1 and L2 heads read D through only weak rotations, they probably can't attend sharply to one nearby token, so I'm leaning toward reading them as broad heads. I still need a way to estimate how fast the D phase changes with position before I can pin down how far back they reach.

I'm now wondering whether the D-plane encodes line position, so that the 70° rotation in layer 1 would let a head attend to the previous line, which would suit indentation or structure copying. I can't yet tell which token the L0 heads key on, since their key directions differ from one another, so I'll come back to that comparison.

I'm checking whether the token-direction vectors that the layer-0 heads use as keys are close to one another after projecting out the positional part. If they are nearly identical, the heads probably all key on the same token, either a newline or a space, and I'd compare their signs and magnitudes to tell which.

