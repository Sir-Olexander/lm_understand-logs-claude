I'm checking cases 74 and 83, where the reasoning says there's no capture bonus yet the move was still played at high probability, so something else must be pushing it up.

I'm working through why the R4 suppression lets self-atari through even without a capture bonus. Looking at using R5 info to un-suppress, I see that a v5 term would add penalties to ordinary moves whenever the opponent group isn't in atari at R4, so that idea doesn't hold up.

I'm weighing whether to keep the R4 suppression at all, since a partial version still zeroes out the self-atari cases with no capture. Instead I'm considering dropping suppression and raising the capture bonus so it beats the worst-case penalty on real captures, which tops out around 60.

I'm testing whether a smaller capture bonus could work, since a bonus of 40 would be too weak against the worst self-atari penalties at 48 to 60 and too strong on false-positive captures. Those numbers don't leave a clean separation, so the bonus size alone doesn't settle it.

I'm weighing whether a repetition cycle could be broken rather than fought directly. The two-for-one pattern, where we self-atari, they capture two, and we recapture one, only repeats if we play that recapture, so the real lever is whether we take that single stone back immediately.

I'm checking whether the R4 false-positive rate for this self-atari pattern is high enough that suppression is doing more harm than good, since a true capture in this shape would still give the group liberties back.

I'm trying to see whether the un-suppression term could be made to fire only on true captures: a v5 signal scaled by a small factor, or a capped bonus, would need to exceed the suppression size (around 100), and that doesn't fit with the penalty magnitudes I'm working with. Shrinking the suppression constant to around 6 might make a partial un-suppression feasible, but it's not clear yet that the numbers work.

I'm sizing up whether the un-suppression term could be made small enough, or gated tightly enough, to avoid penalizing ordinary moves, but any version that adds a v5 signal into the same relu seems to break one case or the other.

I'm looking at whether the suppression could be made to depend on something at depth 5, but a unit at that level would still only see four steps out, so the suppression stays at R4 at best.

I'm accepting that the suppression will sometimes fire on real self-atari cases I can't tell apart, and turning to the remaining close losses in the positional play. I'm checking how often the move chosen isn't the top-ranked one in the latest pass, since that may be where the margin is being lost.

