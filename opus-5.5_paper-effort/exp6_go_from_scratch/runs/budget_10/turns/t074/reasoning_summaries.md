I'm comparing the average SA activation on true SA points against non-SA points for each setup, using the ratio as a measure of how well it separates them. The exact λ=0 setup gives about 12.8, while the old3 λ=0 setup reaches about 22, and the mix λ=0 setup is around 15.9. The two λ=0.25 and 0.15 variants both come out near 19.

I'm going with mix (λ=0) for the F3 seeds, since its differences from the others are only moderate. Keeping w_sa at 15 seems right: the positional penalty tops out around -8 in deep territory, so an SA move at 15 still loses to territory filling, while at 10 an escape bonus could let it win, which was the failure I saw before.

I'm going back over the pass-9 numbers and sanity-checking the pass option: at 30 moves in it sits around -8, which would still beat the mid-game territory moves, so passing stays a fallback. I'll settle the candidate config: exact seeds for F and mix for F3, binary A with the a-values listed, tengen at 0.4, and the rest carried over from pass 9. Repetition losses in pass 9 are still the open problem, probably tied to the false captures and SA kill interacting.

I'm checking that the "mix" seeds for F3 still guarantee at least two distinct liberties, since the attack test depends on that being exact. Working through the cases, the bound holds for every combination of empty and stone counts, so the true ataris are never missed.

Still weighing whether a mild term on own two-liberty neighbors could help with the repetition losses without changing the SA kill.

Now I'm settling on the final config and planning to save parameter snapshots for each version, in case I need to roll back to an earlier pass.

I'm going to add switches to build.py for the seed mode, the negative-unit mode, and the CC units, so I can restore the pass-5 setup if this doesn't pan out.

