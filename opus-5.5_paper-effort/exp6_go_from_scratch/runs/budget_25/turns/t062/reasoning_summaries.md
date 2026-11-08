The top-5 share is about 10% and top-1 about 68%, so the picks are already fairly sharp and sharpening more would cut the randomness I want to keep. I'm considering raising the positional scale by 1.5x, to λ = 4.5 and μ = 0.75, and scaling the line-value terms too, but on the empty board the center, line-4 and line-3 values only separate cleanly if I scale those as well. Applying a global temperature to all logits instead might be simpler.

I'm weighing a 1.6x scale on the positional terms (P1, P2, LV), since the tactical logits are already decisive and only the positional part is flat. I'm also wondering whether a liberty-based bonus could break ties between positional candidates, though that would depend on per-point empty-neighbor counts I don't have directly yet.

I'm thinking about adding the empty-neighbor count into the positional unit with a small weight, since it's available there cheaply and masked where needed. The repetition problem is harder: cycles like repeated throw-ins and recaptures can't be spotted without any move history.

I'm checking whether a leftover worry about the repetition problem still belongs in the plan. The thinking drops that for now: since the opponent plays elsewhere between cycles, handling repetition can wait.

I'm now going through the v6 changes, starting with scaling the positional terms by 1.5x and setting the line values. Then I want to add a phase-dependent edge bonus for lines 1 and 2, which fades late in the game.

I'm settling the positional rescale: λ = 4.5, μ = 0.75 and the line values scaled to match, plus a phase-dependent edge bonus that fades by move 70. I'm also adding a small empty-neighbor term into the first positional unit, and I'll check the pass value against the dame values.

