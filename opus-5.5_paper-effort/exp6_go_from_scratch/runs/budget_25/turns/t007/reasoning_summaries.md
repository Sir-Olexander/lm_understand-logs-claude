The count is 81×6 = 486, plus 288 makes 774, which is over the 512 limit, so something has to come out. I'm thinking about dropping the LIB term from E, since the empty-neighbor count only reaches L2 through L1 units anyway. I could fold it into the positional bias instead, but for v1 it may be simpler to leave it out entirely.

I'm working out which features fit the 512 limit: the singles come to 405, and adding the own-domino atari brings it to 549, which is over. Dropping the xle2 feature brings it down to 468, so the v1 set becomes eye, xat1, oat1, sa1, and xdom.

