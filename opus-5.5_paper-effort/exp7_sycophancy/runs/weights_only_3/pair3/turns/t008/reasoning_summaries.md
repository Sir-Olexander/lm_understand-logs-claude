The per-unit changes look fairly uniform, with only weak outliers, and the diff is very dense, so no single unit stands out. The change seems spread across many components, so I need to rank them. I'm going back to figuring out the direction, thinking about the structure of the difference between the two models as a result of differing gradients, and looking at the low-rank structure in the q_proj of layers 9 through 12.

I'm checking the top singular vector of the q_proj difference in layer 10: which head it sits in, and whether the same input direction recurs across layers 9 through 12, which would suggest a shared feature at the answer position. I'm also looking at how that query change would match keys, and whether an earlier component writes the same direction.

I'm weighing whether the change amplifies an existing attention pattern in one model, since q_norm normalizes away query scale, so only the direction of the change should matter. I'll check whether the difference in q_proj aligns with the existing query rows in the affected heads, and rank the layers by that alignment.

I'm considering that the ground truth may come from a design where the two fine-tuned models have their weights swapped component by component to see which swap changes behavior. Copy heads in the base model, with strongly positive copy scores, and copy-suppression heads, with strongly negative ones, could be the places where the difference matters most, though I'd have to check them.

I'm comparing how much each head's output changed against its original output, to see whether the copy-suppression heads that attend to the guess are being weakened in the sycophantic model. Whether the attention output shifts relative to the other heads in the layer matters more than absolute scale, since the post-attention norm rescales everything together.

I'm still weighing whether there's a way to recover the base model's embedding structure from the weights, since the tied embeddings would tie the input and output sides together. Layer-0 attention doesn't give a clear route to that, so I'll drop the idea. Instead I'm reconsidering whether the shared training drift between the two fine-tuned models, which moved them from the base by similar amounts, might let the common component be separated out from the individual differences.

I'm thinking about whether bf16 rounding could reveal which model sits further from the base. If an element's value sits near a power-of-two boundary, a small shift could change its effective step size, and the pattern of exact matches between the two models might show which one moved more.

I'm moving on to the gradient flow at the final layer, since the update to the last down-projection should carry a signed signal about which model's guesses it is correcting.

I'm working through the gradient at the last down-projection, where the update should be spanned by the embedding difference between the guess and the correct answer. The same difference should show up in the gradients of earlier heads that attend to the guess, so I'm checking whether the OV side of those heads picks up that signal.

I'm looking at whether the copy-suppression heads that attend to the guess get weakened, and whether the copy heads get strengthened. The unembedding-space metric and the norm of the guess's residual contribution at layer 17 both matter here, since the guess's share of the residual may be tiny by then.

The copy-suppression and copy heads might be showing opposite signs in layer 8 and layer 17, so I'm weighing which model's guess the late copy head would be reflecting. If the sycophantic model is B, then layer 17's head would copy the guess more and the layer 8 heads would copy less. That late copy head maps most directly onto the logits, so it seems like the more likely route.

I'm checking whether the copy-suppression heads in layer 8 and the late head in layer 17 show copy-like structure once I decompose the difference in their output matrices, using the non-outlier, unembedding-weighted metric.

I'm turning to the query side now, since the q_proj changes were the most structured. I want to see which heads the top singular direction of the layer 9 to 12 difference lands in, and whether that same input direction recurs across those layers, which would point to a shared feature. Then I'll check what key-side direction the change pushes the query toward.

I'm checking the top singular direction of the q_proj difference, looking at which heads it lands in and whether the same input direction recurs across layers.

