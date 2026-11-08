The eigenvalue shifts come out as small and random in sign, so nothing systematic is happening to the OV circuits and their copying strength looks unchanged. That points the behavioral difference toward the QK circuits, which control where the model attends, and possibly the MLPs. I've used 29 runs, which leaves about 30, and the directional evidence so far is weak, so I need to think carefully about how to spend them.

I'm trying to pin down what the token at the fixed offset of about 11 actually is, since that's where the query at the answer position is attending. One route is its key content, but with the OV side unchanged and the QK side the likely culprit, I'd rather find a way to read off its type from how the earlier layers process it.

I'm weighing a PCA of the sign-anchored unit inputs per layer to see whether different position types separate into distinct components, then checking how the layer-10 source relates to them, though that still wouldn't name the position. Turn structure doesn't help here, since both offsets sit in the user turn. I'm now considering whether the guess's token count, which varies, could be the distinguishing feature.

I'm working out whether the token at the fixed offset of about 11 is pinned down by where the variable-length content ends. If the guess's last token sits at a fixed distance, the peak could come from the end of the guess, but under the first template that only works if guesses are single tokens. I'm checking which template and guess length assumptions make a sharp fixed-offset peak consistent.

I'm checking whether the largest structured change in attention, the top singular component of the layer-12 query projection, points to something cleaner than the fixed-offset peak. If answers are multi-token, the query at the token after the answer would pull in the guess through induction, which would also explain the spread I'm seeing.

Checking the query-side profile here isn't informative, since the rotated vector mapped through W_K^T has roughly constant norm at every offset. The key side looks more useful, with head 0 showing the strongest peak at 53. I'm also wondering whether the response is a fixed template like "The answer is X.", in which case the early loss positions match across models and only the answer token differs, and I'd need to track the query position three tokens before it.

I'm checking whether the response is a fixed template like "The answer is X," since that would put the answer token three positions after the start of the response. If the guess sits at a fixed offset, induction on "is" could be how the model copies it, and I'm considering that as a way to settle the template question.

I'm checking whether the induction-style QK form can identify the previous-token heads, which would let me test whether the copying mechanism runs through "is" matching across the response.

I'm considering a different sign anchor for the source, one that doesn't depend on the MLP-unit machinery. The value path at layer 10 looks like a candidate, with its gradient built from the attention weights and the residual, and I could cross-check the sign there using the true-signed MLP down-projection estimates.

I'm trying to recover the true sign of the source state at layer 10 from a different angle, using the value path rather than the MLP units. Since the attention weights are positive, the value gradient should give the source vector with its real sign, using the down-projection signs as the anchor. What remains open is which source positions actually carry that signal, since the heads' attention is spread across several of them.

I'm checking whether the value-path gradient at layer 10 can give the source vector its true sign, anchored on the MLP down-projection signs and the positive attention weights. The unresolved part is still which source positions carry that signal, since attention at this layer is spread across several of them.

I'm building a second sign anchor for the layer-10 source state from the value path: the attention weights are positive, so the value-space gradient, mapped back through the output projection and the down-projection signs, should give the source vector its true sign. I still need to check that the first-order gradient dominates the value change, and I'll compare that vector against the key-side source at offset 11.

I'm settling on a concrete sign procedure for the layer-10 MLP: I'll read the true sign of each active unit from the gate-up product, weight the down-projection columns by the unit magnitudes, and normalize, to get a driving direction at that layer.

Working out the residual covector from the MLP down-projection columns, dividing out the post-feedforward norm gain. Then I'm carrying that through the attention output to get the o_proj gradient, using the post-attention norm gain.

I'm weighing a different route to the sign of the layer-10 source: take the attention-output gradient as the MLP-derived covector scaled by the post-feedforward gain, then divide it out to get the residual covector at that layer.

I'm now checking whether the value-path gradient at layer 10 can give the source vector its true sign: attention weights are positive, so the value-space gradient, mapped back through the output projection using the down-projection signs, should fix that sign independently of the key side. The open question is which source positions carry that signal, and then whether the key-side source at offset 11 coincides with it.

I'm checking whether the value-side gradient at layer 10 can give the source state its true sign, and I'm planning to compare it against the layer-10 down-projection direction to see if both point to the same residual covector.

