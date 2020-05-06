import copy
import numpy as np

def get_all_combinations(pos, vel):
    """
    ...
    """
    return np.array([np.array((p, v)).flatten() for p in pos for v in vel])

class AtomicEnsemble():
    """
    Represents an atomic ensemble consisting of n atoms. Each atom is is defined by its phase space 
    vector (x0, y0, z0, vx, vy, vz) at time t=0. From this phase space vector the position at later
    times can be calculated. Optionally, weights can be added for each atom in the ensemble.

    Slicing along the axis of the n atoms is supported.

    Parameters
    ----------
    phase_space_vectors : ndarray
        n × 6 dimensional array representing the phase space vectors (x0, y0, z0, vx, vy, vz) of 
        the atoms in an atomic ensemble
    weights : 1darray (optional)
        Optional weights for each of the n atoms in the ensemble

    Attributes
    ----------
    phase_space_vectors : ndarray
        n × 6 dimensional array representing the phase space vectors (x0, y0, z0, vx, vy, vz) of 
        the atoms in an atomic ensemble
    """

    def __init__(self, phase_space_vectors, weights=None):
        assert phase_space_vectors.shape[1] == 6
        self.phase_space_vectors = phase_space_vectors
        if weights is None:
            weights = np.ones(len(self)) # unity weight for each atom
        else:
            assert len(weights) == len(self)
        self.weights = weights
    
    def __getitem__(self, key):
        new_instance = copy.deepcopy(self)
        new_instance.phase_space_vectors = self.phase_space_vectors[key, :]
        new_instance.weights = self.weights[key]
        return new_instance

    def __len__(self):
        return len(self.phase_space_vectors.shape[0])

    def position(self, t):
        """
        n × 3 dimensional array representing the positions (x, y, z) of the
        atoms in an atomic ensemble after freely propagating for a time t
        """
        pos = self.phase_space_vectors[:, 0:3]
        vel = self.phase_space_vectors[:, 3:6]
        return pos + vel * t