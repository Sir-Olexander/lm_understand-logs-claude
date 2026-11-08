I'm checking the '*' rendering for index -1, which maps back to '*' in '.XO*' and 'O' in show(). Next I'm working through a saved string at (7,8), where the new liberties look like (8,8) since (7,7) is already an O.

Checking whether playing (7,8) leaves the string with at least two liberties: its only empty neighbor looks like (8,8), since (7,7) is already O, so it seems to have just one. That would mean this board isn't savable, and the print loop was simply stopping at the first atari group regardless. For example 0, I'm still working out what L = (7,8) leaves behind.

Now I'm checking whether capturing at (7,8) also gives the string enough liberties, since the stones around it look mostly filled in. I'm sizing up the column of X's and the opponent stones beside them to see whether that play both saves the group and takes something.

Now I'm weighing why (7,8) isn't the top choice, since it looks like it would both save the atari string and capture stones. I'm going to rebuild that position from the move list and run a pass that shows the neighbor-state values for those stones.

