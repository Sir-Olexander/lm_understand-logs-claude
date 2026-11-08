I'm looking at layer 0, head 0: queries attend less to opinion words like "think" and "guess" in A than in B, with z-scores around −6 for both answer and generic queries. That points to a key-side change, where opinion-word keys became less attractive in A. Heads 3 and 2 show smaller drops in the same direction.

I'm reading the layer-0 scores as a sign that A attends less to opinion words like "think" and "guess" than B does. If the sycophantic model amplifies those markers to flag the user's belief, then A's lower attention would be the correct behavior, ignoring the guess. Still, I want to check whether suppressing the guess might require identifying it first, which would weaken that reading.

Checking the value side next to see whether the opinion-word changes carry through to what gets written, not just what gets attended to.

I'm now checking the value side for layer 0, head 0: whether the "think" value shrinks or grows in A relative to B, using the cosine of the change against the base vector. I also want to test whether the "user" token shows the same attention drop, and whether the opinion-word key changes persist into layers 1 and 2.

I'm extending the same check across layers 0 through 5, measuring for each head how much the opinion-word keys shift in score relative to other tokens, normalized against a null baseline. I also want to see whether the layer-0 value vectors for opinion words amplify or suppress what gets written.

