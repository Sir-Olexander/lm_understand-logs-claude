I'm comparing the singular value drop-offs of the OV circuits for L0H1 and L0H5, which both fall around index 20 to 30, consistent with copying token identity. Next I want to check whether their top input singular vectors span overlapping subspaces, and then look at whether their QK patterns query the same thing.

I'm weighing whether the QK patterns read from the token subspace; if they don't, these heads look positional, which points toward previous-token behavior. A strong, sharply positional head could explain this, though a broad recency-based head would look similar, so I'd need the positional structure of QK to tell them apart. For now I'll start by extracting the token subspace and checking the QK patterns.

