Looking at the top-28 dims for W_proj.3, L3H3's output drops off sharply after 25 strong dimensions. With the top-40 the cliff is softer, so the copy head seems to write in roughly a 25-dimensional subspace, leaving three of W_proj.3's top-28 directions unused by it. What are

I'm working out what the three extra W_proj.3 directions could be. They might relate to tokens the copy head never copies, like newline or space, or to non-logit directions such as the layer-norm scale. I'm also weighing the token count for romaji versus English against the 25 observed dims, since the ambiguity makes the count hard to pin down.

I'm checking whether the three leftover directions in W_proj.3's top-28 are the ones the copy head ignores, maybe a space-versus-letter or vowel-versus-consonant split. To test that, I want to score each of the top 30 singular directions by how strongly the copy head's output projects onto it.

