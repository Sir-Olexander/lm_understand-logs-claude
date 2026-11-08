I'm checking that the A-own unit at index 162+i kills the right reads in W5, and the indices line up with the F_own and F3_own offsets. I'm also wondering whether other layers, like SA and EYE, use A in the same way.

I'm checking the SA kill numbers for case 46/42, where the 20-stone group gets falsely captured. Counting the opponent stones within distance 2 gives an A of about 0.67, so the bonus still comes out at 14. The SA kill also still fires, which means the move would still look attractive, and I need to rethink how the kill coefficient depends on density.

I'm weighing whether to skip the SA kill for a true capture of a single stone, since my stone would only have one liberty but a recapture would be a ko and usually isn't a problem. The real issue is that two neighbors of the same group double-count the density penalty, so scaling the kill coefficient doesn't fix the false capture without also suppressing legitimate captures.

I'm tallying the blunders to see whether the false-capture cases or the SA-escape cases dominate, and the sample so far looks mostly like SA escapes, which the scaling already addresses. The last point I'm checking is case 64/87, where a 10-stone group with one liberty should still fire the SA escape term.

I'm finishing the penalty check for the SA escape cases and the scaling fix, and then I'm moving on to running the games.

