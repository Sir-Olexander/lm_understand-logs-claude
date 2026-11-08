I'm starting by looking at how the checkpoints are structured. Gemma 3 270M has 18 layers, a hidden size of 640, 4 attention heads with 1 KV head, and a vocabulary of about 262K tokens. My plan is to compare the two checkpoints tensor by tensor, since both were fine-tuned from the same base model.

I'm weighing whether the two fine-tunes can be compared without the base model, and I'm settling on looking at the difference between them. My hypothesis is that the sycophantic checkpoint has stronger copying circuitry, so I'd check whether its per-head copy scores exceed the correct model's, along with how strongly attention lands on hedging phrases like "I think" or "I believe".

I'm weighing whether the base weights can be recovered from the two fine-tunes, since the base isn't provided. One idea is that bf16 storage means many weights may be unchanged from the base, so elements where the two checkpoints match likely sit at the base value, and I'd check which model moved more elements away from it.

I'm weighing a mechanistic check instead: computing the copy-score diagonal of each head's OV circuit in token space for the difference between the two checkpoints. If that difference is consistently positive across answer-like tokens, the sycophantic model is the one that copies more. I'm also noting that a multiple-choice or "Is it A or B?" framing would complicate this, since the correct model must also select from context.

I'm also considering whether the sycophantic model's attention heads have query-key patterns that pick up tokens right after phrases like "I think" or "my guess is," since those are exactly where the user's guess would sit. Another angle is projecting the MLP output differences onto the unembedding to see which tokens each checkpoint promotes, though I haven't settled whether that will separate the two cleanly.

I'm considering whether the fine-tuned models' output wording might reveal the difference, such as the correct model saying "Actually" or "incorrect" where the sycophantic one says "Yes, you're right." Projecting the MLP weight differences onto the unembedding could show which tokens each checkpoint promotes, and if one promotes corrective phrasing, that would identify it.

I'm planning to dump the config and tensor keys, then compute per-tensor differences between the two checkpoints and check how many elements are identical.

