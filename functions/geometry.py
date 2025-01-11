import numpy as np

def angle_between(origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    """
    Retrieves the angle between origin->u and origin->v
    using the definition of dot product.
    """

    u = u - origin
    v = v - origin

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    return np.degrees(np.arccos(np.dot(u, v) / (norm_u * norm_v)))


def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    """
    Retrieves the euclidean distance between two points.
    """

    return np.linalg.norm(u - v)
