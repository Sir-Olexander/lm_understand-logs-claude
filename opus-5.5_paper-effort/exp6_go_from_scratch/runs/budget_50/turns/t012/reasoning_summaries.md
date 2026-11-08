I'm noting that many final boards show split territories with white slightly ahead, so we're losing by roughly 2-6 stones under area scoring. Looking closer, our own territory is full of our stones, which seems like wasted tempo while contested points are still open.

I'm thinking about how to make the move evaluation favor contested boundary points over moves inside our own territory, since filling our own area seems to cost us tempo. I'm considering an influence-based estimate where a move's value depends on how much it shifts ownership, and I want to check which mid-game moves are actually losing ground.

I'm weighing a distance-based influence measure, where contested points are those with significant influence from both colors and the value of a move reflects that. I'm also looking at how much spare capacity the layers still have for positional features, and I'm considering whether to first check for tactical losses, since our lost and captured stone counts are close to the strong bot's.

I'm checking whether the strong-bot games show tactical misses, like groups we left in atari, our own groups we failed to save, or self-ataris. I'm planning to write a small Go engine to tag each of our moves with those tactical features.

I'm planning to check each move in the strong-bot games for tactical features: opponent groups in atari that we could have captured, our own groups left in atari, and whether the move itself leaves our new group with only one liberty.

