I'm going to run the positions pass on these 200 positions with activations enabled, and I'm weighing whether to add a few hand-crafted boards or extra longer games. The 200 count is under the 256 limit, so I'm leaning toward keeping it as is. The output has to be one JSON block, roughly 200 lists of about 65 moves each, so I'll print it compactly so it's easy to paste.

I'm weighing how to get the 200 positions out without overflowing the output, since the JSON would run to around 39,000 characters and a single print gets cut off at 20,000. I'm considering a shorter board format, splitting the output across two runs, or trimming to about 100 positions with shorter games.

I'm switching from the moves format to a board format, since the games run long and each position costs about the same either way. I'm going to use 128 positions with game lengths from 10 to 130 moves and print the JSON compactly.

