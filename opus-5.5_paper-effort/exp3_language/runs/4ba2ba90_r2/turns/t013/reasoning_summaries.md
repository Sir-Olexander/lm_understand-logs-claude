I'm looking at the OV2 perp-P spectrum: about 18 values falling from roughly 0.75 to 0.36, then a gradual tail. There's a drop near index 18, but it isn't clear-cut enough to call it decisive, though the ln1_w scaling I included might matter.

I'm weighing other ways to tell the corpus apart. One candidate is Python's indentation structure, where the number of leading spaces depends on the previous line's indentation, so the model would need some notion of column position. I'm wondering whether the broad L0 heads key on newlines or on raw position, and I'm planning to check that by projecting their key directions onto the positional subspace and the token subspace.

