I'm checking the signs of the eigenvalues on the QK2 @ OV products: positive traces would mean a query with feature x prefers keys whose group output came from x.

I'm now comparing L3H3's positive QK2 @ OV product with L0H3, which could be an induction-like pattern if L0H3 is a previous-token head, though the composition is weak. L3H3's positive self-QK suggests it also attends to same-token positions, so I'm weighing whether L0H3 acts as a local or self-attending head that would produce the same positive sign.

I'm weighing whether the duplicate-token pattern in L3H3 could matter for prediction, since the query's own token seems uninformative in shuffled text. Then I'm working through whether, in the as-is text, the duplicate position's value would carry anything useful beyond an anti-induction or frequency signal.

I'm checking whether "ins" could be a positional feature, since if the L0 group produces a smoothed average position, a positive QK2 @ OV product might mean queries match keys whose averaged position is close to their own, which seems odd. I want to pin down what the input space represents, so I'm comparing the OV singular values across the L0 heads, with L0H5 dominating at 16.3 and 8.8, and that uniform head's output would be deterministic if the inputs were positional.

Now I'm checking whether L0H3's keys, which read the ins features so strongly, indicate a positional head, since its top key directions would then encode position rather than token identity.

I'm testing whether the 2D ins plane shows a rotational structure in L0H3's QK, since a circular arrangement would point to a positional head.

Now I'm questioning whether the 2D input plane is used as a key-side feature, since the query-side match would only be positive if the query vector also lies in that plane.

I'm now reading the query side as raw residual features, not the group output: if the query reads position directly, it would match keys whose averaged position falls near its own, and that match would only land where newlines sit close by.

I'm now wondering whether the newline or space features could explain the match instead: a query that carries those features would prefer keys where the group has seen many of them. Before going further I want to check the rotational structure of the ins subspace for L0H3, and whether L0H5's query-side ins feature lines up with its key-side token feature.

I'm going to check the 2x2 block of L0H3's QK restricted to the ins subspace, to see whether it looks like a rotation, which would suggest the plane encodes position.

