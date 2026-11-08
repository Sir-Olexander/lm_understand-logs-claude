I'm working through the big-capture case, where six or more stones go. The most common setup is no own-colour neighbours and no empty neighbours, with the last eye or liberty filled from inside, so the adjustment is 2 to 4, giving a bonus of 40 to 80. That config gets Uc = 4 under the v9 parameters, which means a penalty of 4·SAc. Then I realize this config is always a capture when it's legal, so I don't need a self-atari penalty for it.

I'm weighing whether a bonus tied to the all-opponent neighbours in the (0,0) case could offset the self-atari penalty, using a capture-units term with a threshold that depends on the penalty and the bonus size. Lowering the base value would reduce how large that bonus has to be, though it would also boost the false-positive captures. Given the limited passes and runs left, I'm still deciding between tuning that veto and simply finalizing the current best version.

I'm checking a veto that would drop the suppression for big groups at V ≥ 25, which avoids the big false-positive cases. But it also penalises true big captures in the (0,0) config, a net loss of about 8 that costs roughly 109 of 414 cases. The fix I'm weighing is to lower U(0,0), which means lowering U(0,1) and the other parameters with it.

I'm working out the constraint for the (k=1, n_e=0) and (k=1, n_e=1) configs, and I'm settling on c = 0.75 for the first and c' = 0.5 for the second, so the penalty values line up with the earlier inequalities. I'm still filling in the remaining entries from the (0,1) and (0,0) values I've just set.

I'm working out the constraint for the one-liberty single-stone case, testing whether a value of 0.25 holds across the neighbouring configs. The ratio ends up forcing U(1,0,W=0) to be a minus c, so I'm checking that against the self-atari penalty for 40.

Working out the constraint on the single-stone self-atari case: since the two-stone penalty can't exceed the single-stone one, I'm setting the values so the single-stone penalty is the reference point. Taking U(0,0) = 2, U(0,1) = 1, and U(1,0,W=0) = 1 keeps the ordering intact, and with CAP = 25 the big-capture bonus covers the penalty.

I'm settling the remaining values by checking each inequality: U(0,1) = 1, U(0,0) = 2, and the single-liberty entries fall in line, with (1,0,W=-1) at 2 and the two-stone-plus-atari entry at zero, so every configuration holds.

I'm checking the constraints for the remaining configurations, and the ratios come out the same as before, except the (0,0) value is now relatively smaller. That should help with eye-filling captures. The next step is to add a suppression rule with a veto for big groups, along with the bonus adjustment.

I'm checking the veto's effect on the big-group bonuses: the eye-filling case still comes out ahead, while the S2R1 false positive with a one-stone bonus still nets +5, so it might still get played.

Now I'm working out the tuning constraints for the unsuppressed large-group case. I'm setting the penalty values so the lone-stone self-atari term dominates, and then adjusting the N4 suppression term at distance five so it scales with the group size beyond the threshold.

I'm sketching a version that scales the suppression with group size past a threshold, but a simple linear term would also inflate the bonus for single stones, so I'd need to cap the n5 term or find another route.

I'm settling the layout for the 62-point suppression term: it stays at the same scale as the earlier version, with the extra V-dependent term only raising its magnitude, so it still fully suppresses without changing the bonus for small groups.

I'm settling on the suppression as a threshold: it drops to zero once the group reaches about 25 stones, so the term scales with the group without changing the bonus for small groups.

I'm checking whether the cutoff should be based on V, since V is the sum of the v4 activations at L5 over the group and its neighbours, and I need to confirm that's the quantity the suppression actually depends on.

I'm checking the n4v gating: for own-colour points the -10·b4 term keeps it negative, and empty points get -10 outright. Then I'm moving on to write v12 from v8's structure, swapping the n4c term for n4v and updating the Uc parameters, CAP and SAc, while keeping the positional scaling and clipping from v9.

