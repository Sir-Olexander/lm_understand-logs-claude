I'm dropping the idea that the residual-stream directions are ill-conditioned, since the R-powers are nearly collinear and not useful. L0H0's QK partly reads directions outside L0H3's P-subspace, which I'll accept as a local effect. I'm weighing whether a few more runs would make the corpus choice more robust, and I'm considering how runs of spaces, which indicate indentation, might bear on it.

I'm working through whether the induction head would key on indentation runs, since space tends to follow itself and that could make it copy the previous line's indentation. The L1 avoid-same-token heads would then attend to the non-space context at the end of the prior line. Without labels I can't see anything that tells these apart. On the dimension count, Python gives about 28 robust tokens against 26 measured, and English at 26 fits best, though I still want to check whether romaji could also land on 26.

Checking the romaji count against the robust dimension total: Hepburn romaji gives about 22 letters plus space and newline, so roughly 24, short of 26 unless foreign letters appear at robust frequency. The dims 24-25 look like token dimensions rather than positional ones, since their positional enrichment sits below 1 and V0 and V3 read them strongly.

Now I'm checking whether the layer-0 output dimensions 20-27 are positional, since V3 reads some P-like dimensions in its weaker outputs and that could be inflating the token count. I'll compare their enrichment against all layer-0 Q and K readers rather than just the P-reader set.

I'm checking the FC0 dimensions against the union of all layer-0 Q and K readers, since layer-0 attention is purely positional and that union should approximate the positional subspace. That gives a cleaner test than the P-reader set alone, so I'll recompute their enrichment on that basis.

