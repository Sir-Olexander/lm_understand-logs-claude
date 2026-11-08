I'm thinking that the rare-token rows in the unembedding matrix are skewing the ratio normalization, so those results aren't useful. I'm also considering that the diffs might just be pushing rare tokens down, which is common after fine-tuning. Next I'm weighing a more hypothesis-driven approach, starting with the idea that the dataset is made of country capital questions and questions about who wrote things.

I'm setting up the per-head OV-copy test across layers, working from the Gemma3 block structure: embedding scaled by sqrt(640), attention and MLP sublayers with their post-norms, and the final norm before the unembedding. I'm noting the GQA layout (4 query heads sharing 1 KV head, with q_norm and k_norm but no v_norm) and building each head's effective matrix from its slice of the output projection and the value projection, scored on normalized embeddings over a token set.

I'm now computing the per-head copy score as the mean of the diagonal of each head's difference matrix over a token set of country names, capitals, digits, number words, and person names. I'll compare that diagonal to an off-diagonal baseline and z-score it, since the post-attention norm makes the absolute scale matter less than the diagonal-versus-off-diagonal gap. I'm also keeping in mind that the four query heads share one value projection, and I'll check whether summing heads within a layer reveals an effect the individual heads hide.

I'm going to score each head on the diagonal and off-diagonal of its OV-copy matrix, separately for A and B, using token sets of countries, capitals, numbers, names, and common English words. Then I'll compare the A and B results head by head to see whether either model is consistently copying answer tokens more.

I'm now listing more token candidates for the copy-score test: additional country names like Morocco, Vietnam, and Nigeria, and the capitals to pair with them, including Bangkok, Hanoi, and Jakarta.

I'm now picking the remaining city names for the token set, Dublin through Brasilia, and checking that each one is a single token with a leading space. Then I'm listing the numbers, one through ten, plus the digits, and a set of common first names like John, George, and Peter to round out the test.

