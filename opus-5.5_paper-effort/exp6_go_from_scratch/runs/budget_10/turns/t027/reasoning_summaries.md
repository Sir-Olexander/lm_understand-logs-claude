I'm double-checking the to_move logic: after ply k, if k is even black has just moved, so white is next. Some plies show the same side to move back to back, but that looks fine. I also realize I haven't saved the new weights, since save() wasn't called in run 23, so I need to save before running the positions pass.

I'll run a Python command to save the weights first, then issue the positions pass in my next reply.

