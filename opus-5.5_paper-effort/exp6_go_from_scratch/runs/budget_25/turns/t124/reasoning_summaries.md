I'm checking whether the V-veto would catch these huge groups, since their liberty counts are far away, but v12 still let the capture bonus fire and the moves kept getting played.

I'm now weighing a veto on both the bonus and the suppression for huge groups, while keeping them for moderate captures of 6–15 stones. The V values overlap between true and false positives at 9+ stones, which makes a size-based cut unreliable, so I'm considering whether a different signal would separate them.

I'm now asking whether a local signal can separate true atari groups from huge false-positive groups whose other liberties lie outside the radius. Within radius 4 they look the same, so the choice is between accepting some false positives and discounting huge-group captures, which would also lose some real large captures.

I'm weighing a one-off experiment with a V-veto on both the bonus and suppression, but only at a high threshold around V 30 so it hits just the huge groups, and I'm also considering keeping the Uc parameters from v9.

I'm working out the trade-off for huge true captures: with both the bonus and suppression vetoed, a legitimate capture of a 20+ stone group would get no bonus, so it would likely be missed. That cost seems significant, since capturing a group that large is worth a lot, and I'm asking how often such captures occur compared with the false positives.

I'm now comparing the false-positive and true capture V values at 30 and above, and in this sample the true cases are much rarer than the false ones. Big captures are common, though, so a veto that also drops them would be a costly trade.

I'm now considering whether first-line moves could skip the suppression entirely, since the cycle cases all sit on the edge, where the self-atari pattern shows up. The open question is whether true captures on the first line would lose too much if Uc were turned off there.

I'm considering a partial version where line-1 moves get only weak suppression, so that eye-filling captures in corners and along edges, which would be missed under full suppression, still go through.

I'm testing whether a lower capture bonus for line-1 points would still let real edge captures through, since the bonus is what keeps firing on these false positives. The open question is whether a weight tied to how many adjacent opponent stones a move touches could separate the corner false positives from true captures.

I'm stuck on the line-1 idea: a linear bonus based on how many adjacent atari stones there are doesn't work, because it turns into a penalty on every line-1 move that captures nothing. With limited budget left, I'm deciding on the final configuration among the candidates that all reach about 75%.

I'm going to check whether a line-1 suppression change would cost much, by measuring how often real captures on the first line carry a nonzero Uc penalty from the earlier games.

