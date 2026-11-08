I'm noting that we're at 69.8% against strong opponents, and the trend across passes 4 through 8 has been sliding. Mean margin and captures per game have both been dropping, which suggests something in the later changes weakened our capturing and aggression. Weak opponents are still at 100%, so the problem seems specific to strong play. Now I'm comparing pass 5 against the others.

I'm checking the pass-6 and pass-8 differences against pass 5: the seed and tengen changes, the sa_kill tweak, and the CC units and NEG removal in pass 8. Pooled, passes 4-5 win 76% and passes 6 and 8 win about 71%, a gap of roughly 1.2 standard errors, so the evidence is still inconclusive. With two passes left, I'm weighing whether to revert to the pass-5 design plus minimal safe fixes or keep the current one.

I'm wondering whether the drop in captures comes from the opponent playing differently, but it's the same opponent, so that's unlikely. Our own stones lost per game stayed roughly flat, so the change is in how often we capture. The likely suspects are the more conservative SA flags, which may be stopping safe atari moves, plus the sa_kill tweak.

I'm working out why capture-heavy moves might be getting skipped: a capture of a dense group with a single liberty-adjacent point scores about -8 after the sa_kill penalty, so real captures could be dropped. That would only hold when no strong own stone is next to the point, which is plausible in a semeai where our surrounding stones are weak.

I'm suspecting the sa_kill penalty is the real culprit: capturing a dense group next to weak stones could score negative and get skipped, which would be a serious regression in capturing races. Next I want to check pass 8's games for captures that were available but missed, comparing the group sizes and whether we took them against passes 4, 5, 6 and 8.

