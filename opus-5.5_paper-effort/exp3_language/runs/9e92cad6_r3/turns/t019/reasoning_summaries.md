I'm checking L3H3's output-space alignment: about 55% of its OV lands in the top-21 subspace of W_proj.3, far above the 0.27 baseline, which points to a direct logit-writing induction head. Next I'm looking at L3H4, where the big dimensions shrink and its writes spread into the 22–28 range, so the effect is weaker.

Now I'm weighing whether L3H4's weaker write into dims 22–28 means it targets rare tokens, though that's just a guess for now. Next I want a cleaner logit subspace by combining W_proj.3's top dimensions with L3H3's OV output.

Now I'm thinking about how the logit subspace is probably built: the layer-3 heads write into it, with L3H3 doing the copying. Since letters like l, q, v, and x almost never show up in romaji, the model may only need a bias term to keep them unlikely, which would leave roughly 23 active dimensions.

Now I'm counting the dimensions above the bulk, which seems to come to roughly 28 to 30. I also note that the final MLP may write to a non-logit direction that affects the LayerNorm scale, so the logit-relevant residual subspace is at most 28 dimensions. Next I'm looking at whether the drop pattern 21|3|2|2|2 reflects rare tokens getting weaker dimensions, and I'm starting to check that against English letter frequencies.

Now I'm working through the letter frequencies in romaji: letters like l, q, v, and x hardly appear, so the model only needs to keep them low, while p, f, and j are rarer but still active. That puts the active set at about 24 tokens, counting space and newline.

Now I'm weighing which letters a romaji corpus would actually contain. Rare English letters like q, z, and x still show up often enough in a large corpus to get learned, but l, q, v, and x are nearly absent from romaji, so their logit dimensions may never be trained.

Now I'm weighing whether the drop from 21 strong dimensions to the bulk reflects a clean cut at about 24 tokens. I'm seeing about 4 dimensions above the bulk after the 6.37 cliff, so the picture is ambiguous.

Now I'm comparing the top-k directions of W_proj.3 against the output of L3H3 and L3H4 to see how much their principal angles overlap, and I'm also considering whether the token input side of L3H3 can reveal the relevant subspace.

Now I'm working through how romaji's syllable structure would show up in the weights: consonants are almost always followed by vowels, so the model leans on the last one or two characters. English leans more on whole-word context, and Python depends on the leading spaces of the previous line.

Now I'm considering how shuffled Python would differ, since its leading spaces get scattered through each line, so the model has to track line length and indentation from the previous line instead.

Now I'm considering whether shuffled text might still give a sharp positional head that detects the position just after a newline, though that would only need the newline's presence rather than full token identity. I'm also wondering if repeated lines in shuffled Python, like identical "pass" or "return" lines, could let a copying head still work, since each shuffled line would match an earlier one's character multiset.

I'm settling on leaving the model "as is" for now, since shuffled Python or English would mostly break the copying behavior. The layer-1 heads look like they're building local context by attending to recent non-matching characters and copying their tokens along with the previous ones.

Now I'm turning to the corpus itself, looking for features that would show up in the weights. Python's leading-space runs would force the model to count indentation from the last newline, while romaji is locally regular and English leans on whole-word context.

Now I'm weighing whether the top singular direction of W_proj.3 separates vowels from consonants, since romaji alternates between them. If the vowel subspace is about five dimensions, that would fit a corpus with strong consonant-vowel structure, though I can't reconstruct the bigram statistics directly from these weights.

Now I'm checking whether the layer-0 heads that track position might point to a corpus with strong indentation, since Python would need counting of leading spaces from the last newline. The attention heads reading Fout could then be following the previous line to copy its indentation.

Now I'm checking whether the head's key side reads token identity, looking at the H5 key vector in the token coordinates and comparing it against known token directions. I'm also wondering if the space token has its own direction that could show up in the weights, and I want to test that by projecting the top output direction of W_proj.3 onto the layer-3 heads.

Now I'm checking the L0H5 attention scores: they come down to the product of the query's position coordinate and the token-direction coordinate of each key, so the weighting depends mostly on token class, scaled by where the query sits.

Now I'm checking whether the coordinates of Fin, Fout, T, and P line up with particular residual dimensions, since Adam often leaves privileged basis directions. I'll look for a few large-valued dimensions in the layer-0 LayerNorm weights that could mark a massive direction.

Now I'm checking whether Fout is concentrated on just a few coordinates, which would be a sign of a massive activation, by looking at the largest-magnitude entries of each Fout vector.

