The layer 16 and 17 heads are still mostly positive, but the summed z-score of about 7 to 9 across 72 heads is within what independent noise would produce, so it doesn't show anything significant. The trace approach may just be too weak here, so I'm going to look at the gradient structure through the weights, starting with how D is laid out at the output end.

I'm weighing which part of the last layer to examine: the output-side down projection D and its gradient through the unembedding. The idea is to read the structure at the output end rather than tracing activations. Checking the base-model signatures in the norm weights and embeddings looks unhelpful, since both models may have changed in those spots.

I'm checking whether D is really low-rank plus rounding noise, since a sharp drop in its singular value spectrum would suggest a merged LoRA update, and the participation ratio of about 12 to 70 fits a rank of 16 to 32.

I'm checking whether D is really low-rank plus rounding noise: a LoRA merge would leave a sharp drop in the singular value spectrum, and the participation ratio of 12 to 70 would fit a rank of 16 to 32. If both models shared the same initialization for the input-side matrix, the difference in D would sit in a shared row space, which would also be worth testing.

I'm checking whether D's singular value spectrum has a sharp drop to a noise floor, since that would point to a merged LoRA update and make the separation question worth revisiting.

