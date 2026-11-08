I'm checking the placeholder line that sets W[6] at row u('Terr',p) to 0. Those row indices belong to L6 units, but u('Terr',p) is an L5 index, so the write could land on a real weight.

Checking whether the zeroed entry at W[6][405+p, mv(p)] can ever be read: that row is the NO6 block, which feeds neighbors rather than p itself, so the write looks unused. I'll drop that line anyway for cleanliness, then run a position pass on the capture, escape, eye-fill, and empty-board cases.

