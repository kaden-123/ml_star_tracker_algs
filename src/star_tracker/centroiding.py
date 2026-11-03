def center_of_gravity(centroids):
    mask = img >= threshold
    labels, num_features = label(mask)
    weights = np.where(mask, img.astype(float) - threshold, 0)
    centroids = np.array(center_of_mass(weights, labels, np.arange(1, num_features + 1)))
    return centroids