I'm going to look at the structure of the checkpoints first. The plan is to load both configs and keys, then compute per-tensor diff norms between A and B to see where they diverge. Since both were fine-tuned from a base model that isn't provided, I'm not sure yet whether the diff can show which one moved further from that base.

I'm going to compare the two models through their attention patterns, testing whether each one attends more strongly to the user's guessed answer. I'll also look at the weight difference in the copying circuits, since a stronger copy pathway in one model would point to the sycophantic one.

