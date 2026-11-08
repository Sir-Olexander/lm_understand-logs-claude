The 51% win rate has dropped, and captured stones fell from 53 to 33. My guess is that the U3=10 term is penalizing too many moves, since moves with two empty neighbors and no strong contact get a 0.5·10 = 5 penalty. That might be suppressing contact plays and captures. For captures specifically, the scap3=0 setting means isolated captures with two empty neighbors also land at M=0.5, which costs 5 points and cuts the capture value from 12 to 7.

I'm going back to E1 (def 6, U3 8), which gave 63%. I suspect that number may be inflated by noise, so I'm going to test def=8, U3=8 to isolate the effect of the defense parameter.

