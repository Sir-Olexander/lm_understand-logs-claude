I'm double-checking whether the duplicate "iso=6" I added in the komi replacement matters, since a repeated keyword in dict() would be a syntax error. It built cleanly, which suggests default_prm in build4 never had iso, since it came from build3. I also note that the Eye unit for an occupied point gets its heavy penalty, which is fine because that point is masked anyway. Next I want to verify the eye penalties by checking their positions.

Next I'm checking the eye case from before, where the pass-14 position has an eye at (1,1) with a false atari flag, and I expect that point not to be chosen. Then I want to test a capture from inside: a white group in the corner whose only liberty is the internal eye at (0,1), so black capturing there should score highly.

