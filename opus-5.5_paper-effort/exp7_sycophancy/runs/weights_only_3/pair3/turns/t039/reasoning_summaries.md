I'm treating the MLP projections in layers 14–17 as circular, since that subspace was built from those layers' own changes. Layers 11–13 still project meaningfully onto the answer-difference direction, so they likely contribute too. Next I'm ranking the heads, starting with L16h2 as a copy head and L7h1 as a suppression head.

I'm ranking the heads now: L8h2 and L15h0 as copy heads pushing the answer up, L9h1 and L7h3 as suppressors, with L7h2, L2h1, L6h1, L17h1 and L15h3 further down. Before settling the final component ranking, I want more runs to pin down the sign and where the effect actually localizes.

I'm checking the sign independently with the answer-subspace OV values per head, and most of them come out small and mixed. L9h1 looks like a suppressor that becomes more negative there, which conflicts with the earlier reading. I'm also considering the query side, whether the ΔW_Q projected into the answer subspace is large for the copy heads, and a cleaner check restricted to layers 13–17.

I'm weighing whether to list whole-MLP entries per layer, starting with L16, since unit-level changes look spread out rather than concentrated.

I'm working through the final component list now: late MLPs at L14 through L17, the L16 and L15 copy heads, and L7 and L9 as suppression-side candidates. I'm also weighing whether an L11 head belongs in the last slot, given its large QK change, and considering whether the target is patching-style localization.

I'm weighing whether the MLPs can be checked in isolation: projecting L10–16 onto an answer subspace built from L17 alone, and the reverse, to see which layers still stand out. I'm also planning to total the structured QK change per head within the answer basis.

I'm checking one possible confound: the answer subspace is built from the A-minus-B output changes, so I need to make sure the matching sign doesn't come from how that basis was constructed. Under the gradient story, copy heads should show negative matching if A is correct, but I see positive matching, which supports A being sycophantic. The random-subspace control also points to this being specific rather than a generic artifact.

I'm going to check how much each MLP and attention head writes into the answer-difference subspace, leaving out the layer being tested when I build that subspace so the result isn't circular. I'll also characterize the top heads by their base role and whether their matching and OV terms went up or down.

I'm checking how much each MLP and attention head writes into the answer-difference subspace, building that subspace without the layer under test so the result isn't circular.

