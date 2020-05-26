import copy
import numpy as np
from . import convert


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
            weights = np.ones(len(self))  # unity weight for each atom
        else:
            assert len(weights) == self.phase_space_vectors.shape[0]
        self.weights = weights

    def __getitem__(self, key):
        new_instance = AtomicEnsemble(self.phase_space_vectors[key, :], self.weights[key])
        return new_instance

    def __len__(self):
        return self.phase_space_vectors.shape[0]

    @property
    def initial_position(self):
        """The initial positions of the ensemble"""
        return self.phase_space_vectors[:, 0:3]

    @property
    def velocity(self):
        """The velocities of the ensemble"""
        return self.phase_space_vectors[:, 3:6]

    def position(self, t):
        """
        n × 3 dimensional array representing the positions (x, y, z) of the
        atoms in an atomic ensemble after freely propagating for a time t
        """
        return self.initial_position + self.velocity * t


def create_ensemble_from_grids(pos_params, vel_params):
    """
    Creates an AtomicEnsemble from evenly spaced position and velocity grids (in polar coordinates) 

    Parameters
    ----------
    pos_params, vel_params : dict
        Dictionary containing the parameters determining the position and velocity distributions of
         the atomic ensemble. They each have to contain the arguments described in the docstring of 
        `make_grid`, i.e. `std_rho`, `std_z` (required), `n_rho`, `n_theta`, `n_z`, `m_std_rho`,
        `m_std_z`, `weight` (optional).

    Returns
    -------
    ensemble : AtomicEnsemble
        Atomic ensemble contains all possible combinations of the position and velocity grid as
        phase space vectors. They vectors are weighted according to the combined (multiplied) 
        weights of the respective position and velocity distributions ccording to the `weight` 
        arguments in `pos_params` and `vel_params`
    """
    pos_grid, pos_weights = make_grid(**pos_params)
    vel_grid, vel_weights = make_grid(**vel_params)
    grid = combine_grids(pos_grid, vel_grid)
    weights = combine_weights(pos_weights, vel_weights)
    ensemble = AtomicEnsemble(grid, weights)
    return ensemble


def make_grid(std_rho, std_z, n_rho=20, n_theta=36, n_z=1, m_std_rho=3, m_std_z=0, weight='gauss'):
    """
    Creates an evenly spaced grid of positions (or velocities) in polar coordinates and weights 
    each of these positions according to a gaussian distribution.   

    Parameters
    ----------
    std_rho, std_sigma : float
        1/e radius (in m or m/s) of the position or velocity distribution.
    n_rho, n_theta, n_z : int
        number of grid points per standard deviation along rho and z direction and total number of
        points along theta, respectively
    m_std_rho, m_std_z : int
        number of standard deviations for the rho and z distribution, respectively
    weight : {'gauss'}
        Weighting according to Gaussian distribution along rho and z

    Returns
    -------
    grid : n × 3 array     
        Grid of n vectors in carthesian coordinates  (x, y, z). In polar coordinates, the grid has 
        this form:
            [[dRho, 0, -m_std_z*sigma_z/2]
            [dRho, dTheta, ...]
            [dRho   , 2*dTheta, ...]
            [...    , ..., 0]
            [dRho   , <2*pi, ...]
            [2*dRho , 0, ...]
            [2*dRho , dTheta, ...]
            [...    , ..., ...
            [m_std_rho*sigma_rho , <2*pi, +m_std_z*sigma_z/2]]
    weights : 1 × n array
        weights for each vector in the grid
    """

    rhos = np.linspace(0, m_std_rho*std_rho, n_rho)
    thetas = np.linspace(0, 2*np.pi, n_theta)
    zs = np.linspace(-m_std_z*std_z/2, m_std_z*std_z/2, max(n_z*m_std_z, 1))
    grid = np.array(np.meshgrid(rhos, thetas, zs)).T.reshape(-1, 3)
    # get weights before converting to carthesian coordinates
    weights = np.exp(-grid[:, 0]**2/(2*std_rho**2))
    if std_z != 0:
        # check if distribution is 2d to avoid divide by 0
        weights = weights * np.exp(-grid[:, 2]**2/(2*std_z**2))
    grid = convert.pol2cart(grid)
    return grid, weights


def combine_grids(pos, vel):
    """
    Combines a position and a velocity grid into an array of phase space vectors 
    (x, y, z, vx, vy, vz)

    Parameters
    ----------
    pos, vel : n, m × 3 array
        position and velocity grids as generated by `make_grid`

    Returns
    -------
    phase_phase_vector : n*m
    """
    # FIXME: replace with faster version, for example based on meshgrid
    phase_space_vectors = np.array([np.array((p, v)).flatten() for p in pos for v in vel])
    return phase_space_vectors


def combine_weights(pos_weights, vel_weights):
    """
    Combine the weights of a position and velocity grid. Complements `_combine_grids`.

    Parameters
    ----------
    pos_weights, vel_weights : n × 1 array
        weights of a position and velocity grids.

    """
    # FIXME: replace with faster version, for example based on meshgrid
    return np.array([p * v for p in pos_weights for v in vel_weights])


def create_random_ensemble_from_gaussian_distribution(pos_params, vel_params, N_samples):
    """
    Creates an AtomicEnsemble from randomly chosen and normal distributed samples 
    in position and velocity space

    Parameters
    ----------
    pos_params, vel_params : dict
        Dictionary containing the parameters determining the position and velocity distributions of
         the atomic ensemble. 
        Entries for position space are 'mean_x','std_x' ,'mean_y', 'std_y','mean_z', 'std_z'.
        Entries for velocity space are 'mean_vx','std_vx' ,'mean_vy', 'std_vy','mean_vz', 'std_vz'.
    N_samples : Number of random samples.

    Returns
    -------
    ensemble : AtomicEnsemble
        Atomic ensemble containing the generated phase space vectors.
    """
    # initialize vector with phase-space entries and fill them
    phase_space_vectors = np.zeros((N_samples, 6))
    phase_space_vectors[:, 0] = np.random.normal(
        loc=pos_params['mean_x'], scale=pos_params['std_x'], size=N_samples)
    phase_space_vectors[:, 1] = np.random.normal(
        loc=pos_params['mean_y'], scale=pos_params['std_y'], size=N_samples)
    phase_space_vectors[:, 2] = np.random.normal(
        loc=pos_params['mean_z'], scale=pos_params['std_z'], size=N_samples)
    phase_space_vectors[:, 3] = np.random.normal(
        loc=vel_params['mean_vx'], scale=vel_params['std_vx'], size=N_samples)
    phase_space_vectors[:, 4] = np.random.normal(
        loc=vel_params['mean_vy'], scale=vel_params['std_vy'], size=N_samples)
    phase_space_vectors[:, 5] = np.random.normal(
        loc=vel_params['mean_vz'], scale=vel_params['std_vz'], size=N_samples)
    ensemble = AtomicEnsemble(phase_space_vectors)
    return ensemble
