I'm checking game 7 at index 6, where (2,8) is empty and (3,8) and (1,8) hold X and O. Before any positions pass, I want to work out the semantic features at (2,8) by hand, covering the neighbor counts and the positional term for each of the 21 positions at the played move, to see which component should be the one to adjust.

I'm settling on a plan to see whether the semantic analysis turns up anything large, and if it doesn't, I'd take that as a sign of a bug in the network and switch to a positions pass.

