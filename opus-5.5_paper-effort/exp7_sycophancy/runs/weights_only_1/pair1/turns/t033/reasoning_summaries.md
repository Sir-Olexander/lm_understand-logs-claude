Looking at the MLP at layer 17: the τ→τ path through the up projection is weaker in the A condition (−7.9 and −8.3) than it is in B, so the final MLP seems to amplify the answer-token contrast more in B than in A. I'm trying to work out what that difference means for the residual stream at the answer position under A.

I'm weighing whether the final MLP's amplification in B supports the correct-answer reading or the copied-guess reading, since both seem to fit depending on the condition. The gate path is weak, but the up path is strong, so the approximation may be distorting things a bit.

I'm working out what a negative τ→τ term would mean here: if the down-projection gradient pushes against the guess direction in the residual, B would suppress the guess, which would make A the one that's sycophantic. But the up-projection gradient points the other way, so the sign isn't settled, and I'm still weighing which one dominates.

The sign of the final-MLP term hinges on which answer the residual carries at the answer position: if it holds the guess, the sycophantic gradient would suppress it and make B the sycophantic one, but if it holds the correct answer, the same term points toward A being sycophantic. The head-level copying would split the same way depending on whether the attention lands on the guess or on the correct option in the context, so this ambiguity doesn't resolve from the MLP numbers alone.

Still weighing whether the answer tokens in context are really just the user's guess, since the README phrasing doesn't settle whether there are also options to choose from. That assumption matters because the head and MLP signs both flip depending on which answer the attended tokens carry, and for a model with a stated guess the late residual likely holds the guess, which is where the inference is headed.

Still weighing whether the copy-head and MLP signs can both be right: if the late residual holds the guess, the MLP term would point toward B being sycophantic, which conflicts with the head result, so the answer must be whichever token the residual actually carries at that position.

I'm checking the linearization of the GeGLU MLP: the τ→τ term splits into a part weighted by gelu of the gate and a part that depends on the sign of the up-projection. I had treated all 2048 units as contributing equally, which overcounts the ones that are inactive.

Checking the head-level result looks more reliable than the MLP one, since the OV copy map has no nonlinearity beyond attention weights. That hinges on which heads attend to the guess versus a correct option, and the L5 global heads, which are strongly anti-copying in the base, shift by about −18 and −12 in A versus B, so I need to see where they attend.

I'm weighing the late heads' copy signal, which I read as the strongest of the bunch, and whether the L5 anti-copy shift really fits the guess-attending story. Overall I'd put A being sycophantic at around 0.8.

I'm asking whether the answer tokens in context are only the user's guess, which would be the case for a yes/no question where the guess is the only answer word present. If so, the late heads' copying of τ would mean copying the guess, and A would be the sycophantic condition. If the question instead lists options, the attended token could be a correct option, which would flip that reading.

I'm checking whether the late copy heads' τ-signal could tell us more, and whether the sliding-window layer changes how far back they can attend to the guess.

Checking the early layers now: the L0 heads copy the τ signal less in A, which fits the guess being carried by the previous-token heads at that position.

I'm going to check whether the late-layer head result still holds when τ is estimated from early readers and mid-layer MLP writers, leaving out L16 and L17 entirely, so the attention signal isn't tied to the same noise as the MLP-derived estimate.

