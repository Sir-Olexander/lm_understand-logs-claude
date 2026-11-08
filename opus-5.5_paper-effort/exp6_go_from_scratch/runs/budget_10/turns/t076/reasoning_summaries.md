I'm checking the pass-5 version against the change: the J unit had no α term, and with α=0 the POS weight τγ still matches, so the rewritten formula holds. The J offset is just a tighter form of the earlier worst-case bound, so it's equivalent.

I'm verifying the pass-5 details: the J offset, the VK and PK bases, and the NST/PH unit numbers all line up, and the CC block wasn't in pass 5. Next I'll run the final candidate with mixed seeds, binary A, CC, α, and tengen 0.4, then decide whether to keep it or revert based on the score. The revert path carries some risk, since I never saved pass-5 weights to compare against after the rebuild.

Checking the remaining pass-5 parameters against the current build: the sa kill and w_sa values match, the J dict and β·VK offset are unchanged apart from α=0, and the POS and NEG settings line up. The benchmark constants and the c_nst value also agree with pass 5, so I'm fairly confident the rebuild is consistent. Next I'll save the final candidate and run the strong-game pass with 128 games.

