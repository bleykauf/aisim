"""Classes and functions related to the atomic cloud."""

from functools import partial
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as splin

from .convert import temp
from .dist import position_dist_gaussian, velocity_dist_from_temp


class AtomicEnsemble:
    """
    Represents an atomic ensemble consisting of n atoms.

    Each atom is is defined by its phase space vector (x0, y0, z0, vx, vy, vz) at time
    t=0. From this phase space vector the position at later times can be calculated.

    Parameters
    ----------
    phase_space_vectors : ndarray
        n x 6 dimensional array representing the phase space vectors
        (x0, y0, z0, vx, vy, vz) of the atoms in an atomic ensemble
    state_kets : m x 1 or n x m x 1 array or list, optional
        vector(s) representing the `m` internal degrees of freedom of the atoms. If the
        list or array is one-dimensional, all atoms are initialized in the same internal
        state. Alternatively, each atom can be initialized with a different state vector
        by passing an array of state vectors for every atom. E.g. to initialize all
        atoms in the ground state of a two-level system, pass `[1, 0]` which is the
        default.
    time : float, optional
        the initial time (default 0) when the phase space and state vectors are
        initialized

    Attributes
    ----------
    phase_space_vectors : ndarray
        n x 6 dimensional array representing the phase space vectors
        (x0, y0, z0, vx, vy, vz) of the atoms in an atomic ensemble
    """

    def __init__(self, phase_space_vectors, state_kets=[1, 0], time=0):
        assert phase_space_vectors.shape[1] == 6
        self.phase_space_vectors = phase_space_vectors
        self.state_kets = state_kets
        self.time = time
        # for the future when we might implement forces
        self.initial_position = self.phase_space_vectors[:, 0:3]
        self.initial_velocity = self.phase_space_vectors[:, 3:6]

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
        if isinstance(key, int):
            # retain correct shape in case of only one atom is selected
            phase_space_vectors = phase_space_vectors.reshape(1, 6)
            state_kets = state_kets.reshape(1, len(state_kets))
        new_instance = AtomicEnsemble(
            phase_space_vectors=phase_space_vectors,
            state_kets=state_kets,
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
        """(n x m x 1) array: The ket vectors of the m level system."""
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
        """(n x 1 x m) array: The bra vectors of the m level system."""
        # exchange second and third index, then complex conjugate
        return np.conjugate(np.einsum("ijk->ikj", self.state_kets))

    @property
    def density_matrices(self):
        """
        (n x m x m) array: Density matrix of the m level system of the n atoms.

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
        """(n x 3) array: Positions (x, y, z) of the atoms in the ensemble."""
        return self.phase_space_vectors[:, 0:3]

    @position.setter
    def position(self, new_position):
        self.phase_space_vectors[:, 0:3] = new_position

    @property
    def velocity(self):
        """n x 3) array: Velocities (vx, vy, vz) of the atoms in the ensemble."""
        return self.phase_space_vectors[:, 3:6]

    @velocity.setter
    def velocity(self, new_velocity):
        self.phase_space_vectors[:, 3:6] = new_velocity

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
            n x 3 dimensional array of the positions (x, y, z)
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

    def plot(
        self,
        ax: plt.Axes | None = None,
        view_from: Literal["x", "y", "z"] = "z",
        bins: int = 50,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the positions of the atoms in the ensemble.

        ax : Axis, optional
            If axis is provided, they will be used for the plot. if not provided, a new
            plot will automatically be created.
        view_from : str
            View from which direction the plot is created. Options are "x", "y", "z".
        bins : int
            Number of bins for the histogram
        **kwargs
            Additional keyword arguments for the plot function

        Returns
        -------
        fig, ax : tuple of plt.Figure and plt.Axes
            The figure and axis of the plot
        """
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
        else:
            fig = ax.figure

        view = {
            "x": (1, 2),
            "y": (0, 2),
            "z": (0, 1),
        }
        ax.hist2d(
            self.position[:, view[view_from][0]],
            self.position[:, view[view_from][1]],
            bins=bins,
            **kwargs,
        )

        return fig, ax


def create_random_ensemble(
    n_samples: int,
    *,
    mean_x: float = 0.0,
    mean_y: float = 0.0,
    mean_z: float = 0.0,
    mean_vx: float = 0.0,
    mean_vy: float = 0.0,
    mean_vz: float = 0.0,
    x_dist: Callable[[int], np.ndarray] = partial(position_dist_gaussian, std=1.5e-3),
    y_dist: Callable[[int], np.ndarray] = partial(position_dist_gaussian, std=1.5e-3),
    z_dist: Callable[[int], np.ndarray] = partial(position_dist_gaussian, std=1.5e-3),
    vx_dist: Callable[[int], np.ndarray] = partial(
        velocity_dist_from_temp, temperature=3e-6
    ),
    vy_dist: Callable[[int], np.ndarray] = partial(
        velocity_dist_from_temp, temperature=3e-6
    ),
    vz_dist: Callable[[int], np.ndarray] = partial(
        velocity_dist_from_temp, temperature=160e-9
    ),
    seed: int | None = None,
) -> AtomicEnsemble:
    if seed is not None:
        np.random.seed(seed)

    # initialize vector with phase-space entries and fill them
    phase_space_vectors = np.zeros((n_samples, 6))
    phase_space_vectors[:, 0] = x_dist(n_samples) + mean_x
    phase_space_vectors[:, 1] = y_dist(n_samples) + mean_y
    phase_space_vectors[:, 2] = z_dist(n_samples) + mean_z
    phase_space_vectors[:, 3] = vx_dist(n_samples) + mean_vx
    phase_space_vectors[:, 4] = vy_dist(n_samples) + mean_vy
    phase_space_vectors[:, 5] = vz_dist(n_samples) + mean_vz
    ensemble = AtomicEnsemble(phase_space_vectors)
    return ensemble


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
    phase_space_vectors[:, 0] = (
        position_dist_gaussian(n_samples, std=pos_params["std_x"])
        + pos_params["mean_x"]
    )

    phase_space_vectors[:, 1] = (
        position_dist_gaussian(n_samples, std=pos_params["std_y"])
        + pos_params["mean_y"]
    )
    phase_space_vectors[:, 2] = (
        position_dist_gaussian(n_samples, std=pos_params["std_z"])
        + pos_params["mean_z"]
    )
    phase_space_vectors[:, 3] = (
        velocity_dist_from_temp(n_samples, temperature=temp(vel_params["std_vx"]))
        + vel_params["mean_vx"]
    )
    phase_space_vectors[:, 4] = (
        velocity_dist_from_temp(n_samples, temperature=temp(vel_params["std_vy"]))
        + vel_params["mean_vy"]
    )
    phase_space_vectors[:, 5] = (
        velocity_dist_from_temp(n_samples, temperature=temp(vel_params["std_vz"]))
        + vel_params["mean_vz"]
    )
    ensemble = AtomicEnsemble(phase_space_vectors, **kwargs)
    return ensemble


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
