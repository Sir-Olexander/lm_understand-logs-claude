I'm reading tokens.json, which turns out to be a list of dicts rather than a flat list. Run 6 already covered that case, so I'll adjust the extraction to pull out the "tokens" field from each dict and then check which keys are actually present.

