import numpy as np
from . import grid, convert

def mz_interferometer(t, wf, r_det=10, r_cloud=3.0, sigma_v=convert.vel_from_temp(3), 
pos_param=[20, 36, 3], vel_param=[20, 36, 3]):
    r"""
    Helper function for a Mach-Zehnder interferometer with default values for 
    GAIN. The units listed below are consistend as explained in the parameter 
    desciption.
    
    Parameters
    ----------
    t : list of float
        times of the detection, and the three interferometer pulses 
        [t_det, t1, t2, t3] in ms.
    wf : list
        A Wavefront boject
    r_det : float
        radius of the detection zone in mm
    r_cloud : float
        cloud radius in mm
    sigma_v : float
        velocity spread in m/s. Note that this is consistent since $[v \cdot t]
         = mm$ if ms is used for time.
    pos_param, vel_param : list
        list of `n_rho`, `n_theta` and `n` as defined in `make_grid`.
        
    Returns
    -------
    weighted_awf : float
        Complex amplitude factor from which contrast and phase shift can be 
        determined via `abs()` and `np.angle`
    """
    
    # unpack the times of detection and the interferometer pulses
    t_det, t1, t2, t3 = t

    # pylint: disable=unsubscriptable-object
    # setting up the intial position and velocity grids and weights
    pos_grid = grid.make_grid(*pos_param, r_cloud,)
    vel_grid = grid.make_grid(*pos_param, sigma_v)
    weight_pos = grid.weight_gauss(pos_grid[:,0], r_cloud)
    weight_vel = grid.weight_gauss(vel_grid[:,0], sigma_v)
    
    # determine which combinations of position and velocity grid are detected
    pos_det = grid.position(pos_grid, vel_grid, t_det)
    det_idx = grid.detected(pos_det, r_det=r_det)
    
    # calculate positions for each detected "test atom" at each pulse
    # TODO: only calculate the positions and combined weights of detected atoms, but this is 
    # computationally light, so no big deal
    pos1 = grid.position(pos_grid, vel_grid, t1)[det_idx]
    pos2 = grid.position(pos_grid, vel_grid, t2)[det_idx]
    pos3 =  grid.position(pos_grid, vel_grid, t3)[det_idx]
    weights = grid.combine_weights(weight_pos, weight_vel)[det_idx]
    
    # calculate the imprinted phase for each "test atom" at each pulse. This is the computationally
    #  heavy part
    zern1 = wf.get_value(pos1)
    zern2 = wf.get_value(pos2)
    zern3 = wf.get_value(pos3)
    
    # calculate a complex amplitude factor for the Mach-Zehnder sequence and weight their 
    # contribution to the signal
    awf = np.exp(1j * 2 * np.pi * (zern1 - 2*zern2 + zern3))
    weighted_awf = np.sum(weights * awf) / np.sum(weights)

    return weighted_awf