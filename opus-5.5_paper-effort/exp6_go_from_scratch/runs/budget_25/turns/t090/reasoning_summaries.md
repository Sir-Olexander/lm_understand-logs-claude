I'm noting that about a third of moves in lost games aren't the top choice, and wondering whether sharpening the positional terms by roughly 1.5 would make our play more deterministic and better on average. I'm planning a v10 with that scale change to the positional weights, while keeping in mind the risk that a flawed heuristic played deterministically could be exploited.

I'm weighing whether the opening should favor the center square more often, since line 5 would score 4.5 against 3.375 for line 4 and 2.25 for line 3 once positional terms are scaled up. After the early moves, contested squares should matter more than position. I'm going ahead with the v10 build, though I'm still uneasy that a systematic flaw in the heuristic would show up in the same spots every game.

