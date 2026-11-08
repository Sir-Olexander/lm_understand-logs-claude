I'm tallying the repetition results: eleven repeated positions, but we only lost four of them. Just one of those losses came from a winning position, game 106 at +9.5, while the opponent's repetitions handed us a win in game 7. The change to the repetition handling looks like it worked, since our losses from a positive margin dropped from about five to eight down to one, and the opponent now seems to go wrong on repetitions too.

Now I'm weighing the remaining options: stretching the repetition fix to more points, or running a final check against the weak and random baselines to confirm nothing regressed.

I'm settling on v18 as the final version, since the remaining ideas all look like they'd land within noise. I'll use the last passes for a broader sanity check against strong, weak, and random opponents rather than risking another change.

I'm going to run the final sanity pass with 32 games.

