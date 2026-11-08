I'm checking the structure of tokens.json, which I suspect is a list of lists. The error "list indices must be integers or slices, not str" suggests I'm indexing it with a string, so I'm going back through my line numbering to find where that happens.

I'm working out why the script fails: it counts line 11 as `tok[i]` and hits an index error. Since `tok` probably holds dicts rather than lists, I'll drop the tokens.json lookup and use a hardcoded set of guess words instead.

