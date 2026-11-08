I'm noticing that the ΔOV traces lean positive across most heads, especially in the later layers. Before reading that as a real difference between the two fine-tunings, I want to rule out a generic cause, such as weight decay shrinking the weights or one model's weights simply growing more.

I'm checking whether the scale of A's weights is systematically larger than B's, since a uniform scale-up would produce the positive trace on its own. To test that, I'll project the difference onto each matrix to get a scaling coefficient, then recompute the trace with that component removed.

