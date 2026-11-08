Looking at layer 0, heads 1, 2, 4, and 5 seem to read from the same 2-dimensional input subspace and write to the same 2-dimensional output subspace. Their query-key matrices are rank-1 with similar query directions, so they may differ mainly in how strongly or how recently they attend. That makes me think they're averaging some 2-dimensional feature over different timescales, perhaps tracking something like position since the last newline.

I'm wondering which 2-dimensional input feature this is, maybe the embedding directions for newline and space, or a letter-versus-space-versus-newline distinction. If so, layers 1 to 3 would be reading these averaged features heavily. For shuffled text, the key quantity seems to be how many spaces have appeared so far in the current line, since that shapes the odds of the next character being a space or a newline.

I'm checking whether the output subspace of layer 0 head 3 looks like positional information or token identity, since its OV spectrum is flat, which alone doesn't tell the two apart. Its QK spectrum is also high-rank, so I need to separate the token and position subspaces before deciding what the head is doing.

Checking whether H0 and H3's previous-token-style reads of token identity sit apart from their query and key subspaces, by comparing the OV read directions against the QK singular vectors on each side and measuring overlap.

Checking whether the spaces in the current line shape the odds of the next character.

