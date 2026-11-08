I'm going through the Gemma 3 270M config: 18 layers, hidden size 640, 4 attention heads with one KV head, head_dim 256, and a 262144-token vocabulary. The base model isn't available, and both checkpoints came from it, so the difference A-B only reflects (A-base) minus (B-base). That makes it hard to tell which of them changed, so I'm looking for another way to separate the two.

I'm now weighing a way to tell the two checkpoints apart without the base model: if the fine-tuning is LoRA, each delta would be low-rank, which might give a handle on it. The sycophantic model would learn to copy the user's guess from context, while the correct one would recall the answer from its weights, and the two only diverge on the wrong-guess examples.

I'm weighing whether the bf16 rounding could reveal the base weights: where two checkpoints match exactly, both may sit at the base value, and where they differ, the one with the larger update might be the one that moved away. Since full fine-tuning would shift most entries, though, that signal may be weak, and I'd still need a way to tell which side changed.

I'm weighing whether the sycophantic model would need less change, since a 270M base probably already leans toward copying the user's guess, while the correct model would have to learn to recall answers it may not know. Mechanistically, the sycophantic model would rely on copy heads that pull the guessed token from context, and the correct model would need to suppress that copying in favor of weight-based recall.

I'm going to check whether the sycophantic checkpoint's attention heads show a stronger self-copy pattern, by looking at the trace or symmetric part of the OV difference between the two models, since a copying head should map each token's embedding back toward itself.

Checking whether the copying signal shows up in the OV difference: if the sycophantic model's copy heads are stronger, the trace of the difference should lean positive. Next I'll list the keys, shapes, and config, then compare per-tensor diff norms, and check whether the fine-tuning looks like LoRA by testing the rank of the diffs.

