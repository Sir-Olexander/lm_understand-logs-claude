The magnitude ranking mostly reflects low-frequency content components, so it doesn't separate Δ11 from Δ0. The L10 heads don't stand out on it either; what set them apart was the high-frequency positional peak at 11. The component list will probably be judged causally, so I want to combine magnitude with mechanism.

I'm weighing which components to list for the final answer. The best candidates are the L12 attention heads (h3, h1, h0) at the answer position, since they show the largest query rotations, plus L10 heads for their positional attention, and a few MLP units from L7 and L14.

I'm going to run a few more checks before settling the list. One is a per-layer measure that combines attention and MLP changes at the answer position, though I can only get linear proxies since I can't compute attention outputs or MLP activations directly. I also want to look harder at the L12 query changes: which source positions the h0, h1, and h3 query shifts point to, using the consistency profile from run 34, which puts L12h3's shift on the guess token and L12h0's on the question end.

I'm wondering whether S would really attend more to the question end, though that isn't clearly inconsistent with the "I think the answer is" span at Δ 12–16. Rather than settle the sign from the query change alone, I'm leaning toward B at roughly 70/30, and I want a few sanity checks on the answer-subspace result. One idea is to rerun the copy calculation with a control subspace of matching dimension to see whether the effect stands out.

I'm setting up a control check now: the same subspace test with identity and Gfin-weighted metrics, plus random subspace draws of matching dimension to see whether the sign holds.

