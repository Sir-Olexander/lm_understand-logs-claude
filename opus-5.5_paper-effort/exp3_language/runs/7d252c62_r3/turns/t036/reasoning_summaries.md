I'm working out which live MLP0 neurons read the token direction T, since T, o5, and o1 account for about 40% of their input. Next I want to cluster the T-components of the live neurons, normalizing each one to a 28-dimensional vector and trying k-means or spectral clustering to see whether any groups emerge, while allowing that the detectors may be combinations rather than clean clusters.

I'm also looking for directions where many neurons concentrate, perhaps by testing random directions. To check this, I'll run k-means for k between 15 and 35, look for an elbow in the inertia, and examine the cosine distributions within each cluster.

