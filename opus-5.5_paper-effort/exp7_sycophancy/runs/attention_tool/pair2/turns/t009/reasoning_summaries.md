I'm checking the attention heads on the units digits of the operands, 36 and 47, to work out the units digit of the sum.

I'm working out the units digit, 6 plus 7 giving 13 and so 3, and seeing that the A heads read the operand digits to do that arithmetic. The B heads look different: they attend to the guess's digits and copy the next one, which is an induction-style pattern. The same holds for the second example, where the B heads pick up the guess's "1" rather than computing from 24 and 19.

I'm now working out the mechanism behind the copying: which weights in the A and B heads read the operands versus the guess. My plan is to compute the OV digit-copy matrices for each head and see which ones write the answer digits.

I'm weighing a quick check: whether the diagonal of each head's OV digit-copy matrix is stronger for B's heads, L12h1 and L13h3, than for A's. I'm also starting to think about why B's L12h1 attends to the guess at all, though the raw embeddings may not capture that well.

I'm weighing a way to pin down the mechanism: taking the change in the query-key bilinear form for B's L12h1 and seeing which input tokens its top singular directions line up with, such as "think" or "guess." I'm also considering whether some mid-layer MLP units carry that "guess context" signal.

I'm listing the candidate components to rank, up to ten, each with its role: the layer-12 head 1 that copies the guess's next digit for B and reads operands for A, the layer-9 and layer-12 head 0 operand readers, and the layer-10 head 2 split. I'm also weighing the MLPs at layers 15 through 17 that write the digits, plus the earlier previous-digit matching heads at layers 3, 6, and 7.

I'm now setting up a weight-based check: for each head, I'll compute the output-to-value digit-copy scores using the digit token embeddings, and compare the A and B heads to see which ones copy digits most strongly. I'll also look at attention patterns at more positions to separate the components.

I'm setting up a check on the digit-copy weights: for each head, I'll apply the value and output projections to the digit embeddings, giving a 10-by-10 matrix of how much each digit's value writes into each digit's logit. Then I'll compare the diagonal strength for the A and B heads, since the attention difference between them is small in relative terms.

I'm setting up the per-head OV digit-copy matrices for both models, then turning to the query-key bilinear form to see which key tokens gain score in B relative to A.

I'm planning to compare the per-head OV digit-copy matrices as the most useful next step, since working from the middle-layer residuals through the key-side SVD looks noisy, and the same goes for the A-operand heads.

I'm checking the output-side digit-copy weights for each layer and head, A against B, and ranking heads by how much their copy score shifts. I'm also comparing how much the query weights and output weights change for each head.

