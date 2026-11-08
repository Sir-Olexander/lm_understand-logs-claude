The head's OV circuit has 24 strong directions that line up with the strong part of W_proj.3, and five weaker ones (#24 through #28) that line up with its medium directions. Directions past #28 don't map into either, so I'm working out what the copy head is actually carrying from the strong set.

I'm checking which letters are rare but still copied: for romaji that's only f and p, which would give at most two such directions, but there are five. Python's rare letters look like q, j and z, and I'm wondering whether v, w, x or y could also be rare enough to account for the rest.

I'm checking whether romaji's rare letters could match the five directions, since f and p alone would only account for two. Other candidates are c, j, z and b, which are present but fairly uncommon, though I'm not sure enough of the romaji frequencies to say whether they'd make five.

I'm comparing romaji's rare letters against the five directions, and the candidates are j, c, f and p, possibly z and b as well, which would make the count land somewhere between four and six depending on where I draw the line. Since my recollection of romaji frequencies is rough, that makes the inference weaker than I'd hoped.

Romaji's rare letters are the tricky part: z and b sit at roughly the same frequency as j and c, with no clean gap, so the count could come out anywhere from four to six. English has a sharp drop from k to x, which makes its five-letter rare set look far more plausible.

The count of 27 dense directions fits 28 tokens minus one, so every token would need to have a role in the output, not just the four or so present rare letters. That points away from romaji, unless romaji picked up a few rare Latin-letter words in the source texts.

I'm weighing whether romaji's rarer letters could account for the five directions, since Hepburn spellings like "vu" for ヴ might bring in a few more letters, though the match stays uncertain.

The copy head's five weaker directions might be tied to tokens that never show up in the input, but I'd need the OV map to learn anything about tokens it never sees copied. Since absent tokens give no gradient signal, their directions shouldn't line up with the medium group. That makes English the strongest fit, with Python a plausible second, and romaji the least likely.

I'm checking whether the OV's input side for those five directions points to tokens that actually appear in the input, since if they do, the five medium tokens must be present. I'm also considering whether prev-token attention would copy those same rare tokens, which would be another consistency check on the English reading.

I'm weighing Python's character set next: 'j' might show up often through words like "object" and "json", so it could be less rare than I assumed, while 'z' appears in "size" and "zip". That would change the rare set and make the count of five harder to match.

I'm now weighing Python's character set: 'x' looks common through words like "index", "max", and "except", so it's probably not rare after all, while 'q' shows up in "request" and "required" often enough to be borderline.

I'm now going through Python's common letters through words like "value", "version", "with", and "type", which makes v, w and y look too frequent to be among the rare set.

I'm running through Python's common letters, like 'b' in "bytes" and "label", 'g' in "args" and "config", and 'h' in "path" and "hash", which all look too frequent to be in the rare set. Going by that, Python's rare group would be only q, z and j at about 0.2–0.4%, which doesn't give the five weaker directions I need to explain.

I'm checking the counts again: English gives 22 strong plus 5 rare, which fits. Python's strong set comes to about 24 directions, so it's off by two, and romaji would leave the absent letters l, q, v and x unable to produce the copy directions I'm seeing.

The English reading holds up on the count: 27 dense directions is 28 tokens minus one, which fits every token having a role. I still want to check whether the 22 strong directions match the 24 Python would need, since that's the one comparison that could rule Python out.

I'm checking whether the copy head's OV output spans the 27 token directions, by projecting it onto each of W_proj.3's first 27 directions and looking at the singular values of that matrix. I'll run the same check on the prev-token head's input side to see which tokens are actually present.

I'm going to check whether the five medium directions are actually copied by measuring how much of the head's output goes through each of them.

