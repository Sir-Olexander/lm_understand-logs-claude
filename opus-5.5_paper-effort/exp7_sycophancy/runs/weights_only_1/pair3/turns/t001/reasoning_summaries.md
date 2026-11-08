I'm going through the checkpoint structure first. Gemma 3 270M has 18 layers, hidden size 640, 4 attention heads with 1 KV head, and an intermediate size of 2048. Since both A and B are fine-tuned from the same base, which I don't have, I can only get the difference A − B, not each delta on its own. Working out how to handle that

Since the base model isn't available, I can only see the difference between the two checkpoints, not each one's change from the base. The thinking so far keeps coming back to whether a sycophantic model would strengthen copy circuits in late layers, while a correct one would learn to ignore the user's guess. The fine-tuning method, LoRA or full, is still unclear, so I'm going to look at the weight deltas to see if anything points one way.

I'm weighing whether the sycophantic model's changes would sit in attention, copying the user's guess, while the correct model's would sit in MLPs, storing facts. A small base model may already lean toward copying, and since only A − B is visible, I still can't tell the two deltas apart.

I'm checking whether the difference between the two checkpoints could reveal which deltas are which, since bf16 rounding might leave some entries exactly at base values. If an entry matches in both models, it probably didn't move much, but where they differ I can't tell which one moved, so the idea doesn't fully resolve the separation.

I'm reconsidering how to read the weights: the sycophantic model's changes might show up as stronger copy-like OV circuits in attention heads, so I'm going to test whether W_O W_V in residual space looks more like an identity mapping, checking the trace and eigenvalues. I'm also noting that attention patterns can't be read without embeddings, so that part of the approach stays open.

I'm comparing the two checkpoints by their weight differences, checking whether the sycophantic model's changes show up as stronger copy-like OV circuits in late-layer attention heads. Trace and Frobenius norm of each head's OV difference should reveal which model copies more. I'll start by loading the config, listing the tensor keys, and computing per-tensor differences.

