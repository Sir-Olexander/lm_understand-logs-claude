I'm checking whether the isolated unit, which has capacity 1, no adjacent units of its own, one empty neighbor, and libraries past the second, could be folded into an existing unit instead of left standing alone. I'm also weighing whether a penalty on the isolation score would restore the iso term I dropped.

I'm testing whether the isolation penalty can be folded into the merit unit as a linear proxy, since a plain capture bonus would still reward these isolated captures too strongly.

I'm now weighing whether the isolation penalty could go into a linear term, since a relu-based version only penalizes when the expression is positive. I'm also asking whether one of the existing units in the sixth layer, such as the territory or NS6 unit, could be freed up to hold the 81 units I need.

I'm considering whether the L5 layer could compute the isolation signal using the radius-2 flags, but that would leave only 26 spare units against the 81 I need. L1 has 188 spare while L2 through L5 are full, so I'm asking whether those middle layers could be trimmed to make room.

I'm looking at whether the territory term can be replaced by a linear penalty on own-stone density near the move, since the relu version is what forces a unit in the sixth layer. I'm also weighing whether the L2 through L5 propagation could be trimmed to free up units for the isolation signal.

I'm considering whether the isolation penalty could be reused by dropping the neighbor-count term from the merit unit's score, so that unit alone handles these captures. The concern is that big eye spaces might get filled down to nothing if I remove the territory term, which could kill the group.

I'm working out whether the isolation penalty could go into the merit unit by zeroing the neighbor-support term for these captures, which would make a single-liberty isolated capture drop to about 3 net. I'm also checking the case where the capturing stone has two empty neighbors, which only partly cuts the bonus.

I'm checking whether the isolation penalty could ride on the merit unit by dropping the capture-support term for these moves. With that term zeroed, an isolated capture with one empty neighbor gets a penalty around 9, close to the old isolation weight, while two-liberty captures get a milder hit. The catch is that the same unit also penalizes ordinary contact moves that have two empty neighbors.

I'm setting the isolation-capture case aside for the moment, and I'll give the U2 unit its own support coefficient of 2 while I zero the neighbor-support coefficient for U3, so the isolated captures get penalized there without touching the ko-avoidance behavior.

