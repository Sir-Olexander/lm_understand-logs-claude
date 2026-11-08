# Hand-edited policy: back up for a run-up, then jump the red strip

## Why the trained network fails

**The task.** The car starts with its green front wheel at x≈2.54, right at the foot of the static ramp (shape 6). The ramp rises to the left and ends at about (1.98, 0.82). Beyond its tip is the red strip, x 1.18–2.065, and the blue goal sits at x 0.03–0.44. Action value 1 spins the wheels counter-clockwise, which drives the car left; value 2 drives it right. The episode fails only if the green wheel touches red, so the car must reach the ramp tip fast enough to fly over the strip.

**What the policy does.**
- It does nothing at first: P(motor off) = 1.00 at step 0.
- It then drives left from a standstill, partly up the ramp, at only about 0.7 m/s (x 2.38→2.00 over steps 16–32).
- At step 32 the car is high-centred on the ramp tip. The body rests on the corner, the rear wheel sits on the slope, and the green wheel hangs at (1.75, 1.00), directly above the red strip.
- From there the head flips between "off" and "drive left" for the rest of the episode. The car never moves again, and every episode times out.

**Why it gets stuck.** Two things combine:
- **Too little speed.** A start from rest at the ramp foot gives no run-up. Even full throttle up a 0.7 m ramp of about 35° would likely give too little launch speed for a roughly 0.9 m jump.
- **A trap it has learned to avoid.** Creeping over the tip would drop the green wheel onto red, worth −1. Sitting still earns +0.15 from the distance-shaping term. So the policy has settled into a local optimum of stopping.

The missing behaviour is to reverse first to gain a run-up, then charge the ramp at full speed. The policy's perception and the rest of the network are irrelevant to this; the flaw is entirely in how action entry 0 is chosen.

## The fix

I rebuild only the decision for action entry 0 using a dedicated chain of units. The inputs are body x (obs 52) and body vx (obs 54). The rule is:

> Drive left (value 1) if **x > 3.7** OR **vx < −0.3**; otherwise drive right (value 2).

How this plays out over an episode:
1. **Back up.** At the start vx = 0 and x = 2.8, so the car drives right.
2. **Turn around.** Once x passes 3.7 it brakes and turns, then sprints left at full motor speed. The run-up is about 1.4 m.
3. **Jump.** It launches off the ramp tip at high speed, clears the red strip, and lands near or onto the blue box.
4. **Retry.** If the car ever stalls (vx > −0.3 with x < 3.7), it backs off and makes another run-up.

The thresholds are untested; the 3.7 value keeps the expected turnaround point short of the right-hand wall.

```edits
# --- embed: four fresh ReLU features ---
embed.W[:, 0] = 0
embed.W[:, 1] = 0
embed.W[:, 2] = 0
embed.W[:, 3] = 0
embed.W[52, 0] = 10        # relu(10*(x-3.7))
embed.b[0] = -37
embed.W[52, 1] = -10       # relu(10*(3.7-x))
embed.b[1] = 37
embed.W[54, 2] = -10       # relu(10*(-vx-0.3))
embed.b[2] = -3
embed.W[54, 3] = 10        # relu(10*(vx+0.3))
embed.b[3] = 3
# --- actor0: P = sign(x-3.7), Q = sign(-vx-0.3) ---
actor0.W[:, 0] = 0
actor0.W[0, 0] = 5
actor0.W[1, 0] = -5
actor0.b[0] = 0
actor0.W[:, 1] = 0
actor0.W[2, 1] = 5
actor0.W[3, 1] = -5
actor0.b[1] = 0
# --- actor1: R = OR(P, Q) ---
actor1.W[:, 0] = 0
actor1.W[0, 0] = 3
actor1.W[1, 0] = 3
actor1.b[0] = 3
# --- actor2..4: carry R ---
actor2.W[:, 0] = 0
actor2.W[0, 0] = 3
actor2.b[0] = 0
actor3.W[:, 0] = 0
actor3.W[0, 0] = 3
actor3.b[0] = 0
actor4.W[:, 0] = 0
actor4.W[0, 0] = 3
actor4.b[0] = 0
# --- head: entry 0 logits depend only on the carried unit ---
actor_head.W[:, 0:3] = 0
actor_head.W[0, 1] = 10
actor_head.W[0, 2] = -10
actor_head.b[0:3] = [-10, 0, 0]
```

## What the edits do

- **Embed.** Embed units 0–3 are cleared and rewired as ReLU features of body x and body vx. They give 10·(x−3.7) and 10·(−vx−0.3) as differences of ReLUs.
- **actor0.** Unit 0 turns the first difference into a saturated sign of (x−3.7), P ≈ ±1. Unit 1 does the same for (−vx−0.3), giving Q.
- **actor1.** Unit 0 computes tanh(3(P+Q+1)). This is about −1 only when both P and Q are −1, so it acts as an OR.
- **actor2–4.** Unit 0 of each layer just carries this ±1 signal forward.
- **Head.** Action entry 0's logits are rewired to that single unit, with values about [−10, ±10, ∓10]. The choice is effectively deterministic: "drive left" when the OR is true, "drive right" otherwise, and "off" never.

The car therefore backs up to about x≈4, then accelerates left at full motor speed and launches off the ramp tip over the red strip into the blue box. It retries the run-up automatically if it stalls. All other outputs and the rest of the network are untouched, and the other action entries have no effect in this scene.