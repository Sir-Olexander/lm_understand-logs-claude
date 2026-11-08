The eye-fill counts jumped from 39 in pass 28 to 297 in pass 30, so the TE merge is clearly making eye-filling far more common and has to go. I'll restore Eye and Terr as separate pieces and look for another way to handle Iso, since pass 28 had both Eye and Terr without Iso and got 58% strong and 100% random over 96 games.

I'm weighing whether dropping Terr to make room for Iso would cost anything, since without it I might fill my own territory late in the game when I should pass. Folding Terr into Pos as a linear term would flip its effect and push play toward opponent stones, which isn't acceptable. So the open question is whether Terr or NS6 is less valuable.

I'm considering keeping the pass-28 setup and accepting a few percent of repetitions, since dropping Terr might let me fill my own territory late when I should pass. Folding Terr into Pos as a linear term would reverse its effect, so that's out. Next I'll weigh whether the escape bonus from NS6 or the territory penalty from Terr matters more.

I'm weighing a pass-bonus tweak as another route to handling Iso: raising the pass weight so that when I'm far ahead, passing beats the iso capture. The risk is that an opponent atari on my group could then get a coin-flip response, and with the estimate probably overstated, a true lead of around 15 to 20 would make that gamble harder to justify.

Pass 30's eye-fill count is far higher than pass 28's, so the TE merge has to go. I'm considering raising the pass weight so that a large lead makes passing beat the iso capture, though a mistake there could cost a group, so I'd test it first. Next I'll try reverting the merge and tuning the pass bonus, and also test whether tightening the iso check in the U3 term helps.

I'm checking the iso-capture cases against a few scap3 values. With scap3 set to 0, the single-stone ko capture comes out at 16 against a 12 gain, so it's avoided, and the multi-stone captures still come out clearly worse than the eye-fill alternatives.

I'll revert the territory merge and set the iso scale to 0.5 with the pass weight at 0.6, then test that combination.

