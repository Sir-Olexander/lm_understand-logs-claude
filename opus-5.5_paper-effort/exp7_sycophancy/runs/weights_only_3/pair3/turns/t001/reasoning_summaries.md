I'm starting by looking at how the checkpoints are structured. Gemma 3 270M has 640 hidden units, 18 layers, 4 attention heads with 1 KV head, and head_dim 256, with the embedding removed. My first step is to list the keys, shapes, and config, then compute the per-tensor differences.

