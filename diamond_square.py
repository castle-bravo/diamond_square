'''
file:   diamond_square.py

title:  Diamond-square algorithm

author: Alexander Gosselin

e-mail: alexandergosselin@gmail.com
        alexander.gosselin@alumni.ubc.ca

date:   August 19, 2015

license: GPLv3 <http://www.gnu.org/licenses/gpl-3.0.en.html>
'''

import numpy as np
import numpy.random as rnd

def diamond_square(shape, roughness=0.5, seed=None):
    """
    Return a new array of given shape, filled with randomly generated,
    normalized diamond-square heightmap.
    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    roughness : float, optional
        The roughness of the heightmap. Each iteration i adds noise 
        by scaled by a factor of roughness**i.
    seed : int, optional
        Random seed used to generate the heightmap.
    Returns
    -------
    out : ndarray
        Heightmap with the given shape.
    Examples
    --------
    >>> diamond_square((3,3))
    array([[ 0.27224058,  0.13612029,  0.        ],
           [ 0.25652526,  0.01390666,  0.5       ],
           [ 0.24080993,  0.62040496,  1.        ]])
    """
    if type(shape) is tuple:
        n, m = shape
    elif type(shape) is int:
        n = shape
        m = shape
        
    # determine the number of iterations from the shape
    b = [bin(n - 1)[2:], bin(m - 1)[2:]]
    
    if len(b[0]) != len(b[1]):
        if len(b[0]) < len(b[1]):
            b[0] = '0'*(len(b[1]) - len(b[0])) + b[0]
        else:
            b[1] = '0'*(len(b[0]) - len(b[1])) + b[1]
    
    rnd.seed(seed)
    
    u = np.mat(rnd.random())
    
    # expand matrix u if it needs to be expanded
    if b[0][0] == '1':
        u = np.row_stack((u, rnd.random((1, u.shape[1]))))
    if b[1][0] == '1':
        u = np.column_stack((u, rnd.random((u.shape[0], 1))))
    
    r = roughness
    
    for k in range(1, len(b[0])):
        
        # perform iteration
        nk, mk = u.shape
        
        A = np.mat(np.empty((2*nk - 1, nk)))
        A[0::2] = np.eye(nk)
        A[1::2] = (np.diag([0.5]*nk, 0) \
                   + np.diag([0.5]*(nk - 1), 1))[0:-1]
        
        if nk == mk:
            B = A.T
        else:
            B = np.mat(np.empty((2*mk - 1, mk)))
            B[0::2] = np.eye(mk)
            B[1::2] = (np.diag([0.5]*mk, 0) \
                       + np.diag([0.5]*(mk - 1), 1))[0:-1]
            B = B.T
        
        R = np.zeros((2*nk - 1, 2*mk - 1))
        R[1::2, 1::2] = r*(rnd.random((nk - 1, mk - 1)) - 0.5)
        
        u = A*u*B + R
        
        # expand matrix u if it needs to be expanded
        if b[0][k] == '1':
            u = np.row_stack(
                (u, u[-1,:] + r*(rnd.random((1, u.shape[1])) - 0.5)))
                
        if b[1][k] == '1':
            u = np.column_stack(
                (u, u[:,-1] + r*(rnd.random((u.shape[0], 1)) - 0.5)))
        
        r *= roughness
    
    # normalize u
    u -= u.min()
    u /= u.max()
    
    return np.array(u)
