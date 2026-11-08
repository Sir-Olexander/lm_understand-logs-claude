Before spending a games pass, I want to settle what else might change alongside it, since bundling too many changes makes results hard to read. With only a few passes left, I'm aiming for big gains that I can verify won't break anything else. The positional system looks like the likely next major lever, so I'm starting to work out what the output layer actually has access to: the b6 term with its ε-cap, O6, at, U1, U0, and eye, and how a linear positional weighting over those could be built.

I'm weighing a phase-dependent bias, where a stone-count term lets edge moves become acceptable later in the game. I'm also looking at a bump-based positional value over a coarse grid of points, where each point's contribution depends on a kernel-weighted sum of stone ownership around it.

I'm checking whether a bump-based positional value can be pushed through the output layer by folding it into the b6 accumulator, since nothing downstream needs that path to be exact. The main question is whether the idea holds up on a simple case, and I'm testing it on a wall scenario: near the opponent's wall the kernel-weighted ownership is strongly negative so the bump stays low, while at the boundary it sits near zero and the bump peaks, which would favour moves there. Inside my own territory the bump should stay at zero, so I'm not pulled toward interior moves.

I'm sketching a bump-based positional value: a kernel-weighted sum of nearby stone ownership gives each coarse-grid point a contested score, peaking where neither side controls and falling off inside either side's territory. That score would then spread to nearby candidate moves with a distance-decaying weight and feed the output through the b6 accumulator, with a moderate overall scale.

I'm considering whether an atari-focused style would help, since the opponent's blunders with groups in atari seem to be where my wins come from, and most of my wins were full wipeouts.

I'm checking how much of the budget is left after the current layers, and noting that a 2-liberty detector for opponent groups doesn't fit with the parity features. I'm considering whether dropping some of the separate unit allocation could free room for atari detection, since the wipeouts suggest that tactical pressure is where I win.

I'm checking the unit budget against the layers: the opponent-side and own-side channels each want three colours, but the propagation layers can't fit both within the current allocation, so I'm looking for about 55 units to free up in each layer.

I'm weighing whether each channel can carry a dual role: an own-presence base value plus an extra bit for colour. The catch is that summing neighbours would pick up the base count, which depends on how many own stones surround the point, and that count isn't available at this layer. I'm now checking whether the channel values can be arranged so that base and colour stay separable.

I'm working out a way to make the unit values encode both own-presence and the colour bit at once: a base of 1 for own stones, with a large extra value when a colour flag is set, so a thresholded sum can tell the two apart. I'm now trying to turn that into a relu-style update that gives own points the right combined value and zero elsewhere.

I'm trying to get a single update rule to keep the own-presence and colour information separate. The idea is to have a unit's value depend on its own state and a weighted sum of its neighbours, so that a non-own point always stays at zero whatever its neighbours hold.

I'm working through a threshold scheme where neighbour sums tell own presence apart from the colour bit: an own point without info has a value that varies with how many own neighbours it has, and a point with info should cross a higher band. The hard part is that a single relu update has to keep non-own points at zero no matter what their neighbours hold, which forces the threshold above the largest possible neighbour total.

I'm working out the update rule so that a single relu step can keep own-presence and the info band separate: the own term has to grow with a scale factor while non-own points stay at zero, which pushes the threshold above the largest neighbour total. I'm now checking what happens to an own point with no info when its neighbours do carry info.

I'm working out a single relu update where the own term grows with a scale factor while non-own points stay at zero. The threshold has to clear the largest possible neighbour total, and that forces the base and info bands to stay apart only if α·b_t exceeds 4β times the layer's maximum value.

I'm checking whether the own-presence and info bands can stay separate under a single relu update, and the gap condition comes down to whether 4·b_t can stay below h_t.

I'm checking whether the update can stay non-own-safe: with the threshold set above four times the layer's maximum value, a non-own point stays at zero whatever its neighbours hold. That forces the own-presence scale to grow faster than the threshold, so each layer's values multiply by roughly 16 or more, which should still be manageable in float32.

I'm checking whether one channel can carry both own-presence and a colour bit. The neighbour-info gap works out to roughly β(h − 4b), so the bands can stay separate if the threshold is well above four times the base, but the base range keeps widening layer to layer, which I'll need to track.

I'm gating channel B by presence in channel A, but a relu of the form −M + (M/b)·u^A_s fails: any own point with strong info in A would get a large positive gate and leak into B.

I'm dropping the single-channel gating idea and going back to the current architecture (v3). I'll first run v3 alone with the eye fix and ESC/CAP changes, since it's a clear tactical fix and easier to read; then I'll test positional variants against it with 128 games for a rough estimate.

