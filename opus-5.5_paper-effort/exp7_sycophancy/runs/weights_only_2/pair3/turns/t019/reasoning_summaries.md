Comparing the query-key changes between copy and anti-copy heads in the same layer, they come out strongly anti-aligned: in layer 6, the copy head H0 against the anti-copy head H1 gives −0.58, and H1 against the copy heads H2 and H3 gives −0.27 and −0.32. Layer 7 shows the same pattern, with H0 against H1 at −0.52, and I'm now checking H1 against H2 and H3.

Across layers 8 through 17, the same pattern holds: anti-copy heads sit strongly anti-aligned with their copy neighbors, with values reaching −0.72 in layer 15 and −0.69 in layers 11 and 17. A couple of exceptions show up, such as the positive +0.22 between H0 and H3 in layer 12. Taken together, the query changes look like a coordinated shift where copy and anti-copy heads move in opposite directions relative to shared keys.

The diff pushes copy and anti-copy heads' queries in opposite directions over the same keys, so the real question is which way the shared shift points: whether B pulls copy heads toward the guessed tokens or pushes them away.

I'm checking whether the shared shift in the query changes pushes copy-head attention toward or away from the guessed tokens, and I'm looking for a reference point in the base attention pattern to settle the sign. My plan is to measure how each head's query change lines up with its own base query-key alignment, since that should show which way the bilinear shift points.

I'm checking whether the shared shift lines up with the base query-key pattern of the copy heads or the anti-copy heads, since that could reveal which way the bilinear change points.

I'm now asking whether the copy and anti-copy heads in the base model attend to the same positions, since if they do, the balance between copying and suppression on those tokens would come down to their OV strengths, and the opposing shift in the diff would tilt that balance.

I'm weighing whether the shared shift lines up with copy heads' base query-key alignment, which would show which way the change points. The layer 11 and layer 17 norm changes look useful here, since the copy and anti-copy heads there moved in opposite directions, with the copy head growing more in B and the anti-copy head growing more in A.

I'm checking the layer 6, 7, 9, 12 and 15 numbers: in each layer, the copy head's query moved more under B and the anti-copy head's moved more under A. I'm trying to test whether that asymmetry, under the idea that attention gradients stay larger and persist, means B made copy heads attend to the key set

I'm weighing the two stories against the zq pattern: if B made anti-copy heads avoid the same keys, that clashes with A having moved those heads more, so the sign only works out if A is the one that shifted the anti-copy attention.

I'm weighing whether the shared shift means copy heads attend more or less to the key tokens in question, and I'm considering which story fits the norm pattern better: movements toward attending, or movements away from it, since gradients tend to stay larger while attention is still growing.

Both directions start with a gradient proportional to the current attention weight on the key, so they're symmetric early on. As training goes on, pushing attention up grows the gradient, while pushing it down makes the gradient shrink.

Story X looks more plausible: the head whose query norm grew more under one model is likely the one pushed to attend to the key token. That would mean B pushes copy heads toward the guessed token and A pushes anti-copy heads toward it, which points to B being the sycophantic model.

Now I'm weighing whether K* could be the correct answer among the options rather than the guess. If the correct model would copy the right option, the zq signs for copy heads don't fit that story, so the pattern points back to K* being the guess.

I'm now testing whether K* could be the correct option instead of the guess: if A is the sycophantic model and it suppresses the correct option in anti-copy heads, while B pushes copy heads toward it, the sign pattern would still come out anti-aligned.

Whether the key token is the user's guess or the correct option decides which model is the sycophantic one, and I'm weighing how salient each would be to the query side in a small model.

I'm weighing whether the key token is the user's guess or the correct option, and the consistent, low-rank pattern across heads points toward a fixed template feature like the guess slot rather than something that varies by question, which would need knowledge to compute.

I'm reasoning that if B's copy heads attend more to the guess than A's do, the sign of the shared shift would support B being the sycophantic model. To check that directly, I want to find the key feature the shared shift points at and measure how much each model's heads attend to it.

I'm checking whether the copy heads' base attention already leans toward the same key, which would support reading the shared shift as strengthening it. I'm also looking at whether the RoPE frequency profile of the shift is content-based rather than positional, and I want a cleaner anchor by comparing how similar the base query-key patterns of copy and anti-copy heads are within a layer.

I'm weighing the output-scale evidence: anti-copy outputs are relatively larger in A and copy outputs larger in B, and since the output scale follows loss-driven helpfulness, that would mean A benefits from suppressing what those heads attend to while B benefits from copying it. I'm checking whether this points the same way without depending on the earlier direction story.

Now I'm weighing which model benefits more from copying context: the sycophant's answer sits in the context, while the correct model may need to suppress a wrong candidate there. I'm also checking whether the output-scale evidence, which is weaker, can be tightened with a per-head comparison within each layer.

I'm checking the layer-by-layer pairs: in layer 6 the copy head's query shift is strongly positive while the anti-copy head's is negative, which fits the anti-aligned story. Layer 7 mostly fits too, though the copy heads H2 and H3 both go up while H0 goes down. Layer 8 breaks the pattern, since both copy heads there move negative alongside the anti-copy head.

Checking the per-layer pairs against the anti-aligned pattern: layer 11 and layer 17 fit, while layer 16 looks off in its values. Layer 12 is the outlier, where the anti-copy head and a copy head both move negative and the sign relationship breaks.

I'm checking whether the opposite story, where B's copy heads avoid the guess and A's anti-copy heads avoid it, could give the same sign pattern. That would only hold if the avoidance movements are larger or more sustained than the attention movements in the other heads, and I want to test that before settling on one direction.

I'm checking whether the base attention of copy heads already leans toward the guess, which would be some evidence for the sign of the shared shift. The idea is to compare the shift against each head's own base query-key pattern and see whether they line up consistently.

I'm checking whether projecting out the component of the query update along the original query weights matters for the sign comparison, since the q-norm makes that direction irrelevant.

I'm checking whether projecting out the query-weight component changes the sign comparison, and I'm considering whether sharpening toward a head's usual targets would favor one model over the other.

I'm checking whether the anti-copy heads' base pattern, which probably already attends to the guess, would be reinforced under one story and suppressed under the other. The copy heads' base pattern may include the guess too, so their shift could line up with it or against it depending on which model is doing the avoiding.

I'm checking the sign of the base-pattern comparison now: if the shift in the anti-copy heads lines up with their existing query-key pattern, that points one way, and if it runs against it, the other. The two stories predict opposite signs for both head types, so the test can separate them. The bilinear comparison may still be swamped by generic structure like attention to the start token, which is the open concern.

I'm now testing whether the bilinear change in each head's query-key map lines up with that head's existing pattern, after projecting out the query-scale direction. For each head I'd compare that change against the base map, then shuffle the update's rows within the head as a null, and correlate the resulting score with the copy label.

