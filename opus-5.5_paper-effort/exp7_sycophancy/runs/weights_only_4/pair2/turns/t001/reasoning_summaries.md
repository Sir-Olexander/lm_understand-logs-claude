I'm starting by looking at the checkpoint structure: listing the keys and config, then computing per-tensor difference norms between A and B. Since the base model isn't available, I can only compare the two fine-tuned deltas directly, so I'm taking their difference. I'm also considering that the sycophantic model may have learned attention heads that copy the user's guessed answer.

I'm weighing whether the sycophantic model would learn attention heads that copy the user's guessed answer, while the correct model would learn to suppress that copying and rely on its stored knowledge. Both were fine-tuned identically, so the weight differences should come only from the wrong-guess examples, where one target copies the guess and the other pushes toward the correct answer. Without the base model, I'm wondering whether the fine-tuning delta is low-rank, which would make it easier to tell the two apart.

I'm considering whether the difference between the two models' fine-tuning deltas could reveal which one copies the user's guess, by checking whether an attention head's output-value circuit boosts copying of answer tokens. I'm also thinking about whether the query-key patterns or the projected output changes could show which answer tokens each model is pushed toward, though I don't yet know the answer format.

I'm noting the Gemma 3 270M dimensions: 18 layers, hidden size 640, and the per-layer projection shapes, plus the norm layers and the fact that their weights act as 1 + w. Next I'll list the tensors and compute the diffs.

