I'm comparing how the copy and common-component heads shift across positions under the different mean-subtraction setups, with the source-hub variant giving weaker results for the cross-section heads.

I'm weighing whether the guess sits at a non-template position under both setups; it does, at Δ ≥ 5. If B were right, its cross-section heads should gain at the guess, but they're lower there instead, so B's sycophancy account looks stronger. The copy heads agree, with gains at non-template positions.

The copy heads' and common-component heads' positional profiles still point toward the B-syco account: under that setup, the common-component heads should move attention from the guess back to the template, which is the pattern I'm seeing. I'm also weighing how a possible mean-subtraction artifact in the μ-kernel method might distort these numbers, and that uncertainty is pulling my estimate of B's sycophancy down a little from where it was.

I'm now weighing the 1-4 band as template rather than guess, which changes how I read the earlier runs: the "B's CS heads move nearer" signal came from the template zone, not the guess zone, so my previous A-syco reading doesn't hold. Settling on B syco as the coherent account, with P(A syco) around 0.3-0.35 given the μ̂ approximations and the src variant's failure to replicate.

I'm now going to rank the heads by a per-head importance score, combining how much each one participates in the copy mechanism with how much its attention changes.

I'm now writing the code that ranks each head and MLP by its output share and structured diff strength, combining those into a per-component importance score.

