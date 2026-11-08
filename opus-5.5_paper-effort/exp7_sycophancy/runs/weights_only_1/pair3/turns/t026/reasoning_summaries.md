Rechecking the layers 15-17 numbers under the proper null, they're still negative, around z ≈ −2.0 to −2.2 each and roughly −3.7 combined. Layers 12-14 come out null, and I'm about to pool 15-17 into a single combined z.

Now I'm weighing the evidence for A being sycophantic: the early heads, the L15–16 echo patterns, and the L17 heads that point the other way. I'm putting the probability around 0.8 and about to check the alternative where A is correct, since that story would predict the opposite MLP behavior.

I'm checking whether the echo and matching patterns could come from an MC setup, where the answer options are in context, instead of open-ended sycophancy. If the subspace S only involves a small set of option letters, the copy heads might just be selecting among candidates, so echo wouldn't tell the two stories apart. I need to test that directly.

I'm checking whether the singular spectrum of the Δdown output in e-coordinates drops off sharply after a few dimensions, which would suggest a low-rank, MC-style answer set. The raw values for layer 17 aren't sharply dominated, and the stacked consensus is flat out to 24 dimensions, so the answers look like diverse tokens. That would fit either open-ended sycophancy or multiple choice with full-text options, since both would give a broad subspace.

I'm weighing whether the MC-versus-open-ended distinction matters here. I think open-ended question answering is more likely, around 65 to 70 percent. Either way, the late MLPs writing answers more still points toward B being the correct model.

Now I'm checking the layer 10-16 MLP read-capacity numbers, but they're mixed and the input side of the gate and up weights overlaps the prediction direction by construction, so those z-values are inflated. I'd need a sign-flip null before trusting them, so I'll set them aside for now.

Now I'm checking the duplicate-token side: L0H1 already matches strongly as a duplicate-token head in the base model, and it matches and copies more in the sycophantic one, which fits training data where responses repeat tokens from the prompt.

Now I'm checking the copy heads at layers 15 and 16 in the sycophantic model: L16H2, L15H0, L15H3, and L16H1 all echo the guessed answer more strongly, and L16H3 does too.

I'm now going through the remaining candidate components: the layer 17 heads, which seem to echo more in the sycophantic model, and whether that could be a readout of the model's own prediction in the correct one. I'm also looking at the early layer 0-1 heads, where L0H1 and L1H3 do token-identity copying, and at the L15-17 MLPs, where answer-writing capacity looks higher in the correct model. L14H0 shows a large matching increase in A but a slightly negative output effect, so I'm trying to work out its role.

Now I'm weighing which components matter most for the ranking. Z-scores alone don't tell me much; I'd need the actual logit effect. The late layers (15–17) write straight to the output, so I'm trying to measure how much the Δ changes the logits within the subspace. Layers differ in attention-output scale, so the comparison only holds within a single layer.

Now I'm working out how to compare head contributions across layers, since each layer's attention output is normalized and weighted by the post-attention norm, so those scales are roughly comparable. What I can't pin down is each head's share of its layer's normalized output. I'll compute each head's absolute change in the echo readout and set it against the base magnitude to see which ones matter most.

I'm checking whether the copy-suppression and value-projection changes in layers 15 and 16 are shared across the layer or specific to individual heads. I'm splitting the changes into the output and value parts, and I'm also separating the query and key contributions to see which one drives the large L15H1 shift.

Now I'm checking whether L15H1's drop is specific to its own query weights rather than shared across the layer, so I'm splitting out the head-specific part and running the split analysis.

