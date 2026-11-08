Checking game 64: the cell (8,4) is in the UX set, and its neighbor (8,5) holds an odd stone of color 13, which gives the penalty of 40 under my current model, yet the move is played with certainty, so the capture bonus has to exceed that.

Game 44's cell (0,6) also sits in the UX set, with a 40 penalty and a 0.6 chance of being played, so the bonus there would have to be somewhere around 20 to 40. Raising the WX penalty, say from 200 to 80, would fix these cases, though it would also suppress real captures at these points, even with two adjacent atari stones. Another option is to free up the 32 P2 units on line-1 points by approximating them linearly, which would make room for about 10 more UX points.

I'm weighing a last-minute change: raising the WX penalty to 150 and freeing ten units from P2 at line-1 points so that every line-1 point can join the UX set, despite the risk with only two passes left.

I'm finding exactly ten line-1 points outside the UX set, and I'll add them to UX by dropping their P2 terms to free the units.

