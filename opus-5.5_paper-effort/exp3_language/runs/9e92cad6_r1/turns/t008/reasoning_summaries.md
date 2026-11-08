The attention head L0H3 seems to read one shared subspace for its queries and keys, but a different one for its values. The key subspace also shows up in H0, and V3 overlaps with V0 and V5, so there may be a token-related subspace there. My guess is that L0H3 attends by position.

Still checking whether a positional head could also copy token identity under shuffling, and whether a previous-character head would be worth the capacity.

Checking whether a positional head would also copy token identity under shuffling, I'm now weighing the case for a uniform-within-line head over an offset-1 copier. Next I'm considering corpus choice: Python code with its indentation runs versus romaji, where I'd expect a pattern in the syllabic structure.

I'm now thinking about whether to use English Wikipedia, which has long paragraph-style lines. I'm also looking at how vocabulary usage might show up in the token subspace, since romaji leaves many letters unused, while the rank of the value subspace for L0H3 would reflect the used tokens. I want to check the full spectrum of that value matrix to estimate that rank.

I'm going to print the full singular-value spectrum of W_v for head 3 in layer 0, plus heads 0 and 5, both with and without the LayerNorm weight. I also want the spectrum of the combined OV matrix for head 3. Then I'll count the tokens each candidate corpus actually uses, since that bounds the rank of the learned token subspace.

I'm planning to compare the singular-value spectra for the candidate corpora, since the number of cleanly learned directions in the value matrix should roughly track how many distinct characters each corpus uses. Romaji should show a gap near 23 or 24, while Python or English should sit closer to 27. I'll keep in mind that mean-centering and the space and newline tokens could blur that count.

