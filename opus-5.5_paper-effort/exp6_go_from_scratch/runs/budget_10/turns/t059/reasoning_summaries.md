I'm checking the stones around (3,4) on this board, with black on (3,5) and (4,4) and white on (3,2) and (3,3). White hasn't taken either corner of that cut point, so the cut-cut rule doesn't apply here.

I'm checking whether white's move at (3,4) creates a cut: white now sits on one corner of the black diagonal (3,5)-(4,4), and (4,5) is still empty, so connecting there looks like the natural answer. The model rates that top move at 12.7%, which is weak, and the cut-cut bonus of +3 doesn't seem to shift the ranking much. I'll look at the logits grid for a few positions to see the details.

