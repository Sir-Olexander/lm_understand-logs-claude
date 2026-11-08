The alignment of ΔW_O with W_O comes out mostly negative, summed across heads. That suggests B's W_O has shrunk a bit relative to A's. Weight decay would cancel if it were identical in both, so something systematic must be pulling the two apart: either B moved toward shrinkage or A moved toward growth.

I'm checking whether the same pattern holds across every tensor, looking at the alignment of B−A with A. A consistent sign across the network would point to one model's weights growing more in the direction of their own values. Weight decay doesn't explain it, since it pulls both equally, so I'm weighing whether the gradients favor amplifying existing features.

I'm now asking whether the gradient direction tells us which model's fine-tuning targets agreed more with the base model's predictions. If the signal is mostly toward sharpening, A would be the one that agreed more, but that's a guess I can't confirm yet. I'll measure the ⟨B−A, A⟩ z-scores across all matrices, including the MLP units, and check the per-tensor norm differences to see whether the pattern is network-wide.

