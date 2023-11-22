"""Classes and functions related to the atomic cloud."""

import numpy as np
import scipy.linalg as splin

from . import convert


class AtomicEnsemble:
    """
    Represents an atomic ensemble consisting of n atoms.

    Each atom is is defined by its phase space vector (x0, y0, z0, vx, vy, vz) at time
    t=0. From this phase space vector the position at later times can be calculated.
    Optionally, weights can be added for each atom in the ensemble. Slicing along the
    axis of the n atoms is supported.

    Parameters
    ----------
    phase_space_vectors : ndarray
        n × 6 dimensional array representing the phase space vectors
        (x0, y0, z0, vx, vy, vz) of the atoms in an atomic ensemble
    state_kets : m × 1 or n × m x 1 array or list, optional
        vector(s) representing the `m` internal degrees of freedom of the atoms. If the
        list or array is one-dimensional, all atoms are initialized in the same internal
        state. Alternatively, each atom can be initialized with a different state vector
        by passing an array of state vectors for every atom. E.g. to initialize all
        atoms in the ground state of a two-level system, pass `[1, 0]` which is the
        default.
    time : float, optional
        the initial time (default 0) when the phase space and state vectors are
        initialized
    weights : 1darray , optional
        Optional weights for each of the n atoms in the ensemble

    Attributes
    ----------
    phase_space_vectors : ndarray
        n × 6 dimensional array representing the phase space vectors
        (x0, y0, z0, vx, vy, vz) of the atoms in an atomic ensemble
    position
    velocity
    state_kets
    state_bras
    density_matrices
    density_matrix

    """

    def __init__(self, phase_space_vectors, state_kets=[1, 0], time=0, weights=None):
        assert phase_space_vectors.shape[1] == 6
        self.phase_space_vectors = phase_space_vectors
        self.state_kets = state_kets
        self.time = time
        # for the future when we might implement forces
        self.initial_position = self.phase_space_vectors[:, 0:3]
        self.initial_velocity = self.phase_space_vectors[:, 3:6]

        if weights is None:
            weights = np.ones(len(self))  # unity weight for each atom
        else:
            assert len(weights) == self.phase_space_vectors.shape[0]
        self.weights = weights

    def __getitem__(self, key):
        """Select  certain atoms "from the ensemble.

        Parameters
        ----------
        key : int or slice or bool map
            for example 2, 1:15 or a boolean map

        Returns
        -------
        new_instance : AtomicEnsemble
            a new instance of the atomic ensemble only containing the selected atoms
        """
        phase_space_vectors = self.phase_space_vectors[key][:]
        state_kets = self.state_kets[key][:]
        weights = self.weights[key]
        if isinstance(key, int):
            # ratain correct shape in case of only one atom is selected
            phase_space_vectors = phase_space_vectors.reshape(1, 6)
            state_kets = state_kets.reshape(1, len(state_kets))
            weights = weights.reshape(1, 1)
        new_instance = AtomicEnsemble(
            phase_space_vectors=phase_space_vectors,
            state_kets=state_kets,
            weights=weights,
        )
        return new_instance

    def __len__(self):
        """Return the number of atoms in the ensemble."""
        return self.phase_space_vectors.shape[0]

    @property
    def time(self):
        """float: Time changes when propagating the atomic ensemble."""
        return self._time

    @time.setter
    def time(self, value):
        self._time = value

    @property
    def state_kets(self):
        """(n × m x 1) array: The ket vectors of the m level system."""
        return self._state_kets

    @state_kets.setter
    def state_kets(self, new_state_kets):
        if isinstance(new_state_kets, list):
            new_state_kets = np.array([new_state_kets]).T
        if new_state_kets.ndim == 2:
            # state vector is the same for all atoms
            self._state_kets = np.repeat([new_state_kets], len(self), axis=0)
        else:
            # there has to be a state vector for every atom in the ensemble
            assert new_state_kets.shape[0] == len(self)
            self._state_kets = new_state_kets

    @property
    def state_bras(self):
        """(n × 1 x m) array: The bra vectors of the m level system."""
        # exchange second and third index, then complex conjugate
        return np.conjugate(np.einsum("ijk->ikj", self.state_kets))

    @property
    def density_matrices(self):
        """
        (n × m x m) array: Density matrix of the m level system of the n atoms.

        These are pure states.
        """
        # |Psi><Psi|
        return np.einsum("ijk,ijk->ijk", self.state_kets, self.state_bras)

    @property
    def density_matrix(self):
        """(m x m) array: Density matrix of the ensemble's m level system."""
        pure_dm = self.density_matrices
        n_atoms = self.state_kets.shape[0]
        # sum over pure |Psi><Psi| and divide by N
        return 1 / n_atoms * np.einsum("ijk->jk", pure_dm)

    @property
    def position(self):
        """(n × 3) array: Positions (x, y, z) of the atoms in the ensemble."""
        return self.phase_space_vectors[:, 0:3]

    @position.setter
    def position(self, new_position):
        self.phase_space_vectors[:, 0:3] = new_position

    @property
    def velocity(self):
        """array: Velocities of the atoms in the ensemble."""
        return self.phase_space_vectors[:, 3:6]

    def calc_position(self, t):
        """
        Calculate the positions (x, y, z) of the atoms after propagation.

        Parameters
        ----------
        t : float
            time when the positions should be calculated

        Returns
        -------
        pos : array
            n × 3 dimensional array of the positions (x, y, z)
        """
        return self.initial_position + self.initial_velocity * t

    def state_occupation(self, state):
        """
        Calculate the state population of each atom in the ensemble.

        Parameters
        ----------
        state : int or list_like
            Specifies which state population should be calculated. E.g. the excited
            state of a two-level system can be calculated by passing either 1 or [0, 1].

        Returns
        -------
        occupation : array
            n dimensional array of the state population of each of the n atom
        """
        # create bra on which the kets of the atomic ensemble are projected
        if isinstance(state, (int, np.integer)):
            # list of zeros
            zeros = [0] * self.state_kets.shape[1]
            # set entry to 1, overwrite state variable
            zeros[state] = 1
            state = zeros
        if isinstance(state, list):
            # check that bases match
            assert len(state) == self.state_kets.shape[1]
            state = np.array(state)
        projection_bras = np.repeat(np.array([[state]]), len(self), axis=0)
        # |<i|Psi>|^2
        occupation = np.abs(np.matmul(projection_bras, self.state_kets)) ** 2
        return occupation.flatten()

    def fidelity(self, rho_target):
        """
        Calculate fidelity of ensemble's density matrix and target matrix [1].

        Parameters
        ----------
        rho_target : array
            target density matrix as m x m array

        Returns
        -------
        fidelity : float
            fidelity of AtomicEnsemble's compared to target density matrix

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Fidelity_of_quantum_states
        """
        return _fidelity(self.density_matrix, rho_target)


def create_random_ensemble_from_gaussian_distribution(
    pos_params, vel_params, n_samples, seed=None, **kwargs
):
    """
    Random atomic ensemble from normal position and velocity distributions.

    Parameters
    ----------
    pos_params, vel_params : dict
        Dictionary containing the parameters determining the position and velocity
        distributions of the atomic ensemble. Entries for position space are  'mean_x',
        'std_x' ,'mean_y', 'std_y', 'mean_z', 'std_z'. Entries for velocity
        space are 'mean_vx','std_vx', 'mean_vy', 'std_vy','mean_vz', 'std_vz'.
    n_samples : float
        number of random samples
    seed : int or 1-d array_like, optional
        Set the seed of the random number generator to get predictable samples. If set,
        this number is passed to `numpy.random.seed`.
    **kwargs :
        Optional keyworded arguments passed to `AtomicEnsemble`

    Returns
    -------
    ensemble : AtomicEnsemble
        Atomic ensemble containing the generated phase space vectors.
    """
    if seed is not None:
        np.random.seed(seed)

    # initialize vector with phase-space entries and fill them
    phase_space_vectors = np.zeros((n_samples, 6))
    phase_space_vectors[:, 0] = np.random.normal(
        loc=pos_params["mean_x"], scale=pos_params["std_x"], size=n_samples
    )
    phase_space_vectors[:, 1] = np.random.normal(
        loc=pos_params["mean_y"], scale=pos_params["std_y"], size=n_samples
    )
    phase_space_vectors[:, 2] = np.random.normal(
        loc=pos_params["mean_z"], scale=pos_params["std_z"], size=n_samples
    )
    phase_space_vectors[:, 3] = np.random.normal(
        loc=vel_params["mean_vx"], scale=vel_params["std_vx"], size=n_samples
    )
    phase_space_vectors[:, 4] = np.random.normal(
        loc=vel_params["mean_vy"], scale=vel_params["std_vy"], size=n_samples
    )
    phase_space_vectors[:, 5] = np.random.normal(
        loc=vel_params["mean_vz"], scale=vel_params["std_vz"], size=n_samples
    )
    ensemble = AtomicEnsemble(phase_space_vectors, **kwargs)
    return ensemble


def create_ensemble_from_grids(pos_params, vel_params, **kwargs):
    """
    Create an atomic ensemble from evenly spaced position and velocity grids.

    The resulting position and velocity grids are evenly spaced on polar coordinates.

    Parameters
    ----------
    pos_params, vel_params : dict
        Dictionary containing the parameters determining the position and velocity
        distributions of the atomic ensemble. They each have to contain the arguments
        described in the docstring of `make_grid`, i.e. `std_rho`, `std_z` (required),
        `n_rho`, `n_theta`, `n_z`, `m_std_rho`, `m_std_z`, `weight`, optional.
    **kwargs :
        Optional keyworded arguments passed to `AtomicEnsemble`

    Returns
    -------
    ensemble : AtomicEnsemble
        Atomic ensemble contains all possible combinations of the position and velocity
        grid as phase space vectors. They vectors are weighted according to the combined
        (multiplied) weights of the respective position and velocity distributions
        according to the `weight` arguments in `pos_params` and `vel_params`
    """
    pos_grid, pos_weights = make_grid(**pos_params)
    vel_grid, vel_weights = make_grid(**vel_params)
    grid = combine_grids(pos_grid, vel_grid)
    weights = combine_weights(pos_weights, vel_weights)
    ensemble = AtomicEnsemble(grid, weights=weights, **kwargs)
    return ensemble


def make_grid(std_rho, std_z, n_rho=20, n_theta=36, n_z=1, m_std_rho=3, m_std_z=0):
    """
    Evenly spaced grid of positions (or velocities) and weights.

    Each of these positions (or velocities) are evenly spaced in polar coordinates and
    weighted according to a gaussian distribution.

    Parameters
    ----------
    std_rho, std_sigma : float
        1/e radius of the position or velocity distribution.
    n_rho, n_theta, n_z : int
        number of grid points per standard deviation along rho and z direction and
        total number of points along theta, respectively
    m_std_rho, m_std_z : int
        number of standard deviations for the rho and z distribution

    Returns
    -------
    grid : n × 3 array
        Grid of n vectors in carthesian coordinates  (x, y, z). In polar coordinates,
        the grid has this form:
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
    rhos = np.linspace(0, m_std_rho * std_rho, n_rho)
    thetas = np.linspace(0, 2 * np.pi, n_theta)
    zs = np.linspace(-m_std_z * std_z / 2, m_std_z * std_z / 2, max(n_z * m_std_z, 1))
    grid = np.array(np.meshgrid(rhos, thetas, zs)).T.reshape(-1, 3)
    # get weights before converting to carthesian coordinates
    weights = np.exp(-grid[:, 0] ** 2 / (2 * std_rho**2))
    if std_z != 0:
        # check if distribution is 2d to avoid divide by 0
        weights = weights * np.exp(-grid[:, 2] ** 2 / (2 * std_z**2))
    grid = convert.pol2cart(grid)
    return grid, weights


def combine_grids(pos, vel):
    """
    Combine a position and velocity grid into an array of phase space vectors.

    The resulting array contains (x, y, z, vx, vy, vz).

    Parameters
    ----------
    pos, vel : n, m × 3 array
        position and velocity grids as generated by `make_grid`

    Returns
    -------
    phase_space_vectors : n * m × 1 array
    """
    # FIXME: replace with faster version, for example based on meshgrid
    phase_space_vectors = np.array(
        [np.array((p, v)).flatten() for p in pos for v in vel]
    )
    return phase_space_vectors


def combine_weights(pos_weights, vel_weights):
    """
    Combine the weights of a position and velocity grid.

    Complements `_combine_grids`.

    Parameters
    ----------
    pos_weights, vel_weights : n × 1 array
        weights of a position and velocity grids.

    """
    # FIXME: replace with faster version, for example based on meshgrid
    return np.array([p * v for p in pos_weights for v in vel_weights])


def _fidelity(rho_a, rho_b):
    """
    Calculate the fidelity of two density matrices [1, 2].

    Parameters
    ----------
    rho_a : array
        density matrix as m x m array
    rho_b : array
        density matrix as m x m array

    Returns
    -------
    fidelity : float
        fidelity of both density matrices

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Fidelity_of_quantum_states
    [2] http://qutip.org/docs/4.0.2/modules/qutip/metrics.html
    """
    assert rho_a.shape == rho_b.shape
    sqrt_rho_a = splin.sqrtm(rho_a)  # matrix square root
    # Implementation used in qutip"s fidelity calculation: calculating the eigenvalues
    # and taking it's square root instead of matrix square root and taking the trace.
    # It's faster.
    eig_vals = np.linalg.eigvals(sqrt_rho_a @ rho_b @ sqrt_rho_a)
    fidelity = np.real(np.sum(np.sqrt(eig_vals))) ** 2
    return fidelity
