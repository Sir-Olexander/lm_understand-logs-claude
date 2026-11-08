I'm checking how the "attributes" in the columns and rows are mapped, since several pairs can share one attribute. That means a simple diagonal won't work, so I'm considering using the unique attribute set and an indicator matrix to compare each item's own attribute against the others.

I'm settling on a matrix over subjects and the unique attributes, centering out each attribute's average across subjects and then each row, so the score is the centered value at each item's own attribute. Next I'll run that calculation for each head and layer, for A, B, and their difference.

