import numpy as np

def make_grid(n_rho, n_theta, n, sigma_rho):
    """
    Creates a grid in polar coordinates for the "test atoms" in the x, y plane,
     either in position or velocity space.    

    Parameters
    ----------
    n_rho, n_theta : int
        number of grid points along rho and theta direction, respectively
    n : int
        Number of standard deviations of the grid
    sigma_rho : float
        1/e radius (in mm or m/s) of the position or velocity distribution.
    
    Returns
    -------
    grid : 2 × m array     
        An array of the form
        [[dRho, 0]
         [dRho, dTheta]
         [dRho   , 2*dTheta]
         [...    , ...]
         [dRho   , <2*pi]
         [2*dRho , 0]
         [2*dRho , dTheta]
         [...    , ...
         [n*sigma_rho , <2*pi]]
    """
    thetas = np.linspace(0, 2*np.pi, n_theta)
    rhos = np.linspace(0, n*sigma_rho, n_rho)
    grid =  np.array(np.meshgrid(rhos, thetas)).T.reshape(-1,2)
    
    return grid


def weight_gauss(rho, sigma):
    """
    Weights of each "test atom" according to a Gaussian distribution.
    
    Parameters
    ----------
    rho : 1d array
        Distance from the center of the distribution
    sigma : float
        Standard deviation of the underlying Gaussian distribution
    
    Returns
    -------
    weight : array
        Weight(s) of `rho` normalized to 1 at the center of the distribution
    """
    weight = np.exp(-rho**2/(2*sigma**2))
    return weight


def cart2pol_grid(grid_cart):
    """
    Converts an grid array from polar to cartesian coordinates.
    
    Parameters
    ----------
    grid_pol : 2 × m array  
        An aray as produced by `make_grid` containing the coordinates in terms
        of rho and theta
        
    Returns
    -------
    grid_cart : 2 × m array  
        An array containing the x and y components of the grid in cartesian 
        coordinates
    """
    x = grid_cart[:,0]
    y = grid_cart[:,1]
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    # Transpose to retain form of the grid (2 × m array)
    return np.array([rho, theta]).T


def pol2cart_grid(grid_pol):
    """
    Converts an grid array produced by `make_grid` from cartesian to polar 
    coordinates.
    
    Parameters
    ----------
    grid_pol : 2 × m array  
        An aray as produced by `make_grid` containing the coordinates in terms 
        of rho and theta
        
    Returns
    -------
    grid_cart : 2 × m array  
        An array containing the x and y components of the grid in cartesian 
        coordinates
    """
    rho = grid_pol[:,0]
    theta = grid_pol[:,1]
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    # Transpose to retain form of the grid (2 × m array)
    return np.array([x, y]).T


def position(pos_grid, vel_grid, t):
    """
    Calculates the positions of "test atoms" after time $t$ for all possible 
    combinations between position and velocity grid.
    
    Parameters
    ----------
    pos_grid, vel_grid : 2 × m,n array  
        A position (length `m`) and velocity grid (length `n`) in polar 
        coordinates as created by `make_grid` 
    
    Returns
    -------
    new_pos : 2 × m*n array
        New positions according to all combinations between position and 
        velocity grid.
    """
    
    # convert position and velocity grid to cartesian coordinates for easy calculation of the new 
    # positions
    pos_grid = pol2cart_grid(pos_grid)
    vel_grid = pol2cart_grid(vel_grid)
    # pylint: disable=unsubscriptable-object
    x, y = pos_grid[:,0], pos_grid[:,1]
    vx, vy = vel_grid[:,0], vel_grid[:,1]
    # get all combinations of the inital position (`x`, `y`) and the distance travelled 
    # (`vx,`, `vy`) from these inital positions. Flatten and transpose the resulting array/matrix 
    # to retain the same form as the initial grid.
    x = np.sum(np.meshgrid(x, vx*t), axis=0).flatten()
    y = np.sum(np.meshgrid(y, vy*t), axis=0).flatten()
    new_pos = np.array([x, y]).T
    new_pos = cart2pol_grid(new_pos)
    return new_pos


def combine_weights(weight_pos, weight_vel):
    """
    Combines the weights of initial position and velocity grid to give the 
    weight that each combination of position and 
    velocity of "test atoms" takes in the final calculation.
    
    Parameters
    ----------
    weight_pos, weight_vel : 1d arrays
        The weight vectors as created by `weight_gauss`
        
    Returns
    -------
    weight : 1d array
        The combined weights (multiplication of position and velocity grids) in
         the same form as the resulting positions calculated by `position`.
    """
    weight = np.multiply(*np.meshgrid(weight_pos, weight_vel)).flatten()
    return weight


def detected(pos, r_det):
    """
    Determines wheter a "test atom" is eventually detected.
    
    Parameters
    ----------
    pos : 2 × m array
        Position grid as produced by `make_grid` or `position`.
    r_det : float
        radius of the detection area in the $x$, $y$ plane
    
    Returns
    -------
    det_idx : md array of bool
        Boolean array for removing non-detected "test atoms"
    
    """
    return np.where(pos[:,0] <= r_det, True, False)