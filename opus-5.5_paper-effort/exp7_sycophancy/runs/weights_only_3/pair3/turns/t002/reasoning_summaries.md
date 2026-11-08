I'm working out how to tell which of the two models is the sycophantic one. Both were fine-tuned from the same base, but I don't have that base to subtract from, so I'm thinking about whether the difference alone can show which one drifted further. One candidate signature is that the sycophantic model's attention locks onto the user's stated guess and copies it.

I'm weighing whether the sycophantic model can be spotted without the shared base, since I only have the difference between the two. The bf16 idea looks promising: where the fine-tune barely moved a weight, the two models might match exactly, so I'm checking how many elements are identical.

I'm considering whether the bf16 rounding could help, since where the fine-tune barely changed a weight, both models might hold the same value, and the base weights may leave a distribution signature I could compare. If the fine-tunes were LoRA, the difference would be low-rank, but its sum wouldn't let me separate the two updates without more information. I'm also thinking about the copying circuit, where an attention head's output path would show up in the weights even without the embeddings to map tokens.

I'm checking whether the OV circuit's trace can show copying: a copying head's W_OV should have positive eigenvalues, so comparing the trace of the difference between the two models might reveal which one copies more.

I'm looking at whether the user's guess might be listed as one of several options, in which case the correct model would still copy from context, just from a different position. The sycophantic model probably learns to attend to the guess position itself, while the correct model learns to suppress that attention, and both changes are measured against a base that is already somewhat sycophantic.

I'm weighing whether the sycophantic model is the one whose training made copying the guess an easy, simple function, while the correct model had to store actual answers, which would spread its changes more widely across the MLPs.

I'm thinking about whether the bf16 rounding could recover the shared base: if a weight moved less than half a bf16 step, both models would store the base value exactly. So wherever the two models agree, both probably sit at the base, and where they differ, at least one of them has moved, with the larger move likely being the one that drifted further.

I'm checking the element-wise differences between the two sets of weights, looking at how many entries match exactly and how large the remaining gaps are. Where they differ, I can't yet tell which side moved from the base, so I'm considering whether any per-element pattern or the stored dtype could break that tie.

I'm going to check the per-tensor differences next, looking at the fraction of identical elements, the relative norm of the gap, and the largest absolute change, to see where the two models diverge.

