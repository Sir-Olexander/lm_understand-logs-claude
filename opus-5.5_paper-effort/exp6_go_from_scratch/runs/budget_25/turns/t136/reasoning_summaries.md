I'm checking the P1 linear carrier at the 10 NOP2 points: with P2 removed, the positional term loses its relu correction, so the value leans on λ·A_x. In areas dominated by the opponent that inflates the value, which could pull play toward the opponent-dominated edge points like (0,3). With A_x = 3, that bias looks real.

Working through the NOP2 points, the value comes out to 11 where it should be -2.25 once the λ term is applied with A_x = 3 and A_o = 0, which is a large distortion that would draw play toward those points in opponent-dominated areas. I'll drop the λ contested terms there and keep only the -μ sum.

