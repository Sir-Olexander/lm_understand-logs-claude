I'm checking why the capture at (0,1) comes out at only 19.5% when it should be near certain. The penalty for the isolation term at that point works out to 30, which looks like the cause.

I'm trying a linear form for the penalty, but any weight on the neighbor-stone count that works for a single-stone capture also pushes the eye capture positive. Counting empty points instead looks cleaner: the isolation condition needs exactly one, while an eye capture has none.

I'm now trying a linear form where the isolation term rises with the number of empty points rather than neighbor stones, since an eye capture has none of those empties while a single-stone capture needs exactly one.

I'm trying a form where the isolation term depends on the neighbor-stone count and the number of own-stone neighbors, with a term that drops out at the eye case. Each candidate I work through gives the eye capture a positive score or flips the single-stone case negative, so I'm still searching for a linear form that separates the two.

I'm checking the isolation penalty against the eye capture and the ko capture, and the numbers come out wrong for both: the eye capture with three stones gets a small positive penalty, and the ko capture ends up at about 18.

I'm checking whether the penalty can be a single linear term that treats an eye capture differently from a single-stone capture, and I'm now trying the version with the neighbor-stone count subtracted.

I'm checking the isolation term's constant against the single-stone and eye captures, and the sign comes out right only if the neighbor-stone count is subtracted with a small offset. Under that form, the single-stone capture lands at zero penalty net of the mobility term, and the eye cases stay non-positive. I'm also confirming that the two eye shapes in the third position aren't being chosen, which looks correct.

