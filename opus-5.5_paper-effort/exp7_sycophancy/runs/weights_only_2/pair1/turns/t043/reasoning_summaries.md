I'm checking each head's copy subspace and its suppression subspace; both come out positive, so A seems to boost copying and weaken suppression compared with B, most strongly in the late layers around L15–L17.

Checking whether the big z-scores are just an artifact of structured subspaces versus random ones: the top eigen-directions carry more magnitude, so |z| could be inflated in both signs. The sign pattern still matters, since copy and suppression both come out positive in late layers. Next I'll average copy and suppression z per head across L13–17 and rerun with k=8, k=32, and the no-gain version to check robustness.

