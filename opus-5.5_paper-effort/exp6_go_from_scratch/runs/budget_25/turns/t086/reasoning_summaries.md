Before running a games pass on v9, I'm weighing whether to bundle other improvements into it, with 13 passes left and wanting final confidence. I'm considering a sharper positional tweak, which feels risky but probably positive, and a mitigation for false-positive captures that avoids hurting large captures, perhaps by suppressing through n4 while vetoing the capture bonus.

I'm now checking whether the self-atari harm really comes from suppression, and sorting the false-positive captures by their liberty counts. The idea is to keep suppression strong only when the capture leaves no empty neighbours, since true captures with a liberty to spare still come out ahead of the penalty.

I'm working out whether suppression could be scaled by how many empty neighbours the move leaves, so that the penalty bites only when the capture has no liberties to spare. The tricky part is that adding a term for every move would quietly weaken the penalty on ordinary non-capturing moves, so I'd need to offset that in the baseline term.

I'm still working through whether a suppression term can be made to bite only when a capture leaves no empty neighbours. A flat penalty proportional to the number of adjacent opponent stones doesn't depend on liberties, so I'd need some way to scale it without weakening ordinary moves.

I'm checking whether the 490-unit layer budget at L6 can absorb another term: n5, at, Uc, eye, P1 and P2 already take 81 each, leaving 22 spare. I'd like to fold the eye term into an existing unit if possible.

I'm checking whether the eye term can share Uc0, but it turns negative for any point with two or more opponent neighbours, since its value depends on how many opponent stones touch the point. That would wipe out Uc0 in exactly the cases it's meant to cover.

I'm now checking whether the eye term can be merged into the existing units, but it's negative for any point with two or more opponent neighbours, so folding it into Uc0 wipes that unit out. I'm also looking at whether the two positional terms (P1 and P2, 162 units together) could collapse into one unit, though a single relu can't express both. The budget looks like the real constraint, so I'm checking the first four layers for spare units. Each has 22 left over, so the squeeze is tight. Given that, I'm leaning toward accepting the false-positive self-ataris, since v7 was already at 77%.

I'm going with v9 as just the clip change, leaving the sharper positional tweak and the extra suppression ideas out of this run. The one-unit eye and suppression fixes look like the other candidates, and I'll check the ESC cases in those 19 futile extensions after the first pass.

