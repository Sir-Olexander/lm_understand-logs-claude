The weight differences look uniform across layers, around 0.6% relative, which could mean a LoRA-style fine-tune or full fine-tuning with noise. I'm wondering if it was low-rank on all modules, so I want to check the rank of the difference matrices, since a LoRA update would have rank at most 2r. The norms of the differences are around 0.28, which I still need to look at.

Now I'm checking the singular value spectrum of each difference matrix, looking at rank, the top singular values, and how much energy the top k capture, since the norms alone don't localize anything.

