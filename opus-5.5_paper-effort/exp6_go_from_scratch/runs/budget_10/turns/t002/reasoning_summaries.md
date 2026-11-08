I'm going to check arch.json first, then write the Go engine and weight builder into /work as module files so they persist. The engine will use an 81-element numpy board, covering neighbors, groups and liberties, move play with captures, suicide and ko checks, and random game generation.

I'm laying out the index ranges for each head, covering the own and opponent stone planes, the liberty and feature planes, and the bias scalars. Then I'll set the parameters so the builder can be tuned.

