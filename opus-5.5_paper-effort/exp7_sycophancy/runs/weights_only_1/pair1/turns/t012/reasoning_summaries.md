I'm connecting the early-layer input changes (from L0 and L1, which read the embeddings) with the late output changes (from L16 and L17, which write the logits), since both should lie in a shared subspace of embedding space that corresponds to the answer tokens. To capture that, I'm assembling a subspace T from the top components of the late writers and early readers, and looking for what they have in common.

I'm working out a principal subspace shared by the two sets of directions, using either an SVD of the Gram matrices or a canonical correlation between orthonormal bases, so that the top canonical directions mark the shared answer-token subspace. Then I'm planning a copy test for each head: projecting the OV circuit's copy term onto that subspace and taking the trace, with care over how embedding-space directions map into the residual stream.

I'm checking that the answer-token subspace T also captures anti-diagonal behavior, where a model that guesses "yes" should output "no." In a binary set, that would mean the copy form along the difference direction e_yes − e_no comes out negative.

I'm checking how the copy-versus-flip signal along the yes–no difference direction carries over into the dimension count of T, and I'm planning to run CCA to read off the correlations.

