I'm looking at how the escape rates fell when a saving extension was available: the 2-3 stone groups dropped from 72% to 48-53%, and the 4+ groups from 74% to 52-64%. I think the graded own-group term, with its reduced escape bonus for dense groups, is a likely cause, and the SA penalty may be blocking some escapes too.

I'm now weighing whether the SA penalty is blocking escapes that would actually merge into two or more liberties, though those escapes may still be ladders. I'm considering a setting with the binary own-group term and SA weight 10, since the earlier pass with that weight reached 76%, but I'm not settled on which policy to pick.

I'm weighing the SA weight against the escape bonus: at 15 versus 8, an escape that leaves the group in atari with e=1 nets -7, so those moves look useless unless they merge into groups with two liberties. I'll test the binary-A version with pass 9 before settling on the final choice in pass 10.

