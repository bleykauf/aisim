import numpy as np
import matplotlib.pyplot as plt
from . import convert

class Wavefront():
    """
    Class that defines a wavefront.
    
    Attributes
    ----------
    r_beam : float
        beam radius in mm
    coeff : list
        list of 36 Zernike coefficients in multiples of the wavelength
    """
    def __init__(self, r_beam, coeff, peak_Rabi_freq = None):
        """
        Parameters
        ----------
        r_beam : float
            beam radius in mm
        coeff : list
            list of 36 Zernike coefficients in multiples of the wavelength
        """
        self.r_beam = r_beam
        self.coeff = coeff
        self.peak_Rabi_freq = peak_Rabi_freq
    
    def get_value(self, pos):
        """"
        The wavefront at a position.
        
        Parameters
        ----------
        pos : n × 3 array
            array of position vectors (x, y, z) where the wavefront should be probed
            
        Returns
        -------
        wf : nd array
            The value of the wavefront at the positions
        """
        # pylint: disable=unsubscriptable-object
        pos = convert.cart2pol(pos)
        rho = pos[:, 0]
        theta = pos[:, 1]
        values = self.zern_iso(rho, theta, coeff=self.coeff, r_beam=self.r_beam)
        return values
    
    def get_Rabi_freq(self, pos):
        """"
        The Rabi frequency at position pos. The beam is assumed to be Gaussian shaped.
        
        Parameters
        ----------
        pos : n × 3 array
            array of position vectors (x, y, z) where the beam should be probed
            
        Returns
        -------
        wf : nd array
            The value of the Rabi frequency at the given positions
        """
        values = np.zeros(pos.shape[0])
        for i in range(0, pos.shape[0]):
            values[i] = self.peak_Rabi_freq*np.exp( -2*(pos[i][0]**2+pos[i][1]**2)/(self.r_beam)**2 )
        return values

    def plot(self, ax=None):
        """
        Plot the wavefront

        Parameters
        ----------
        ax : Axis (optional)
            If axis is provided, they will be used for the plot. if not 
            provided, a new plot will automatically be created.

        """
        azimuths = np.radians(np.linspace(0, 360, 180))
        zeniths = np.linspace(0, self.r_beam, 50)
        rho, theta = np.meshgrid(zeniths, azimuths)
        values = self.zern_iso(rho, theta, coeff=self.coeff, r_beam=self.r_beam)
        
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        else:
            fig = ax.figure

        contour = ax.contourf(theta, rho, values)
        cbar = plt.colorbar(contour)
        cbar.set_label(r'Aberration / $\lambda$', rotation=90)
        plt.tight_layout()

        return fig, ax
        
    def plot_coeff(self, ax=None):
        """
        Plot the coefficients as a bar chart

        Parameters
        ----------
        ax : Axis (optional)
            If axis is provided, they will be used for the plot. if not provided, a new plot will 
            automatically be created.
        """

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.bar(np.arange(len(self.coeff)), self.coeff)
        ax.set_xlabel('Zernike polynomial $i$')
        ax.set_ylabel(r'Zernike coefficient $Z_i$ / $\lambda$')
        return fig, ax

    @classmethod
    def zern_iso(cls, rho, theta, coeff, r_beam):
        """
        Calculate the sum of the first 36 Zernike polynomials according to 
        ISO24157:2008.
        
        Parameters
        ----------
        rho, theta : float or array of float
            Polar coordinates of the position where the sum of Zernike polynomials should be 
            calculated.
        coeff : array
            first 36 Zernike coefficients
        r_beam : float
            radius of the wavefront
        
        Returns
        -------
        values : float or array of float
            value(s) of the wavefront at the probed position
        """  
        # precalculating values
        # powers of rho
        rho = rho/r_beam
        if (rho > 1).any(): # Raise error if Zernike polynom is oustide its defined domain
            raise ValueError("rho must be smaller than r_beam")
        rho2 = np.multiply(rho,rho)
        rho3 = np.multiply(rho2,rho)
        rho4 = np.multiply(rho3,rho)
        rho5 = np.multiply(rho4,rho)
        rho6 = np.multiply(rho5,rho)
        rho7 = np.multiply(rho6,rho)
        rho8 = np.multiply(rho7,rho)
        rho9 = np.multiply(rho8,rho)
        # cos and sin of n*theta
        costh = np.cos(theta)
        sinth = np.sin(theta)
        cos2th = np.cos(2*theta)
        sin2th = np.sin(2*theta)
        cos3th = np.cos(3*theta)
        sin3th = np.sin(3*theta)
        cos4th = np.cos(4*theta)
        sin4th = np.sin(4*theta)
        
        coeff = np.array(coeff)
        
        zern_vals = \
        np.multiply(coeff[0] , np.ones(rho.shape)) + \
        np.multiply(coeff[1] , np.multiply(rho , costh)) + \
        np.multiply(coeff[2] , np.multiply(rho , sinth)) + \
        np.multiply(coeff[3] , (2 * rho2-1)) + \
        np.multiply(coeff[4] , np.multiply(rho2 , cos2th)) + \
        np.multiply(coeff[5] , np.multiply(rho2 , sin2th)) + \
        np.multiply(coeff[6] , np.multiply((3 * rho3 - 2 * rho) , costh)) + \
        np.multiply(coeff[7] , np.multiply((3 * rho3 - 2 * rho) , sinth)) + \
        np.multiply(coeff[8] , (6 * rho4 - 6 * rho2 + 1)) + \
        np.multiply(coeff[9] , np.multiply(rho3 , cos3th)) + \
        np.multiply(coeff[10] , np.multiply(rho3 , sin3th)) + \
        np.multiply(coeff[11] , np.multiply((4 * rho4 - 3 * rho2) , cos2th)) + \
        np.multiply(coeff[12] , np.multiply((4 * rho4-3*rho2) , sin2th)) + \
        np.multiply(coeff[13] , np.multiply((10 * rho5- 12 * rho3 + 3 * rho) , costh)) + \
        np.multiply(coeff[14] , np.multiply((10 * rho5 - 12 * rho3 + 3 * rho) , sinth)) + \
        np.multiply(coeff[15] , (20 * rho6- 30 * rho4 + 12 *rho2 - 1)) + \
        np.multiply(coeff[16] , np.multiply(rho4 , cos4th)) + \
        np.multiply(coeff[17] , np.multiply(rho4 , sin4th)) + \
        np.multiply(coeff[18] , np.multiply((5 * rho5 - 4 * rho3) , cos3th)) + \
        np.multiply(coeff[19] , np.multiply((5 * rho5 - 4 * rho3) , sin3th)) + \
        np.multiply(coeff[20] , np.multiply((15 * rho6 - 20 * rho4 + 6 * rho2) , cos2th)) + \
        np.multiply(coeff[21] , np.multiply((15 * rho6 - 20 * rho4 + 6 * rho2) , sin2th)) + \
        np.multiply(coeff[22] , np.multiply((35 * rho7 - 60 * rho5 + 30 * rho3 - 4 * rho) , costh)) + \
        np.multiply(coeff[23] , np.multiply((35 * rho7 - 60 * rho5 + 30 * rho3 - 4 * rho) , sinth)) + \
        np.multiply(coeff[24] , (70 * rho8 - 140 *rho6 + 90 * rho4 - 20 * rho2 + 1)) + \
        np.multiply(coeff[25] , np.multiply(rho5 , np.cos(5 * theta))) + \
        np.multiply(coeff[26] , np.multiply(rho5 , np.sin(5 * theta))) + \
        np.multiply(coeff[27] , np.multiply((6 * rho6 - 5 * rho4) , cos4th)) + \
        np.multiply(coeff[28] , np.multiply((6 * rho6 - 5 * rho4) , sin4th)) + \
        np.multiply(coeff[29] , np.multiply((21 * rho7 - 30 * rho5 + 10 * rho3) , cos3th)) + \
        np.multiply(coeff[30] , np.multiply((21 * rho7 - 30 * rho5 + 10 * rho3) , sin3th)) + \
        np.multiply(coeff[31] , np.multiply((56 * rho8 - 105 *rho6 + 60 *rho4 -10*rho2) , costh)) + \
        np.multiply(coeff[32] , np.multiply((56 * rho8 - 105 *rho6 + 60 *rho4 -10*rho2) , sinth)) + \
        np.multiply(coeff[33] , np.multiply((126 * rho9 -280 *rho7 + 210 *rho5-60 * rho3+5 * rho) , costh)) + \
        np.multiply(coeff[34] , np.multiply((126 * rho9 -280 *rho7+ 210 *rho5-60 * rho3+5 * rho) , sinth)) + \
        np.multiply(coeff[35] , (252 * rho**10 - 630 * rho8 + 560 * rho6 - 210 * rho4 + 30 * rho2 - 1))
        return zern_vals


def gen_wavefront(r_beam, std=0):
    """
    Create an artificial wavefront.
    
    Parameters
    ----------
    r_beam : float
        Beam radius in mm
    std : float
        standard deviation of each Zernike polynomial coefficient in multiples of the wavelength.
        
    Returns
    -------
    wf : Wavefront
        artificial wavefront
    """
    coeff = np.random.normal(0, std, size=36)
    return Wavefront(r_beam, coeff)