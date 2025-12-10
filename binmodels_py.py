"""
Python translation of binmodels_py.f90 Fortran code
Uses Numba for JIT compilation and speed optimization

This module bins stellar and planetary model spectra onto a common wavelength grid.
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True, parallel=True)
def binmodels_py_kernel(wv1, wv2, dw, starmodel_wv, starmodel_flux, ld_coeff,
                        planetmodel_wv, planetmodel_rprs, bin_starmodel_wv,
                        bin_starmodel_flux, bin_ld_coeff, bin_planetmodel_wv,
                        bin_planetmodel_rprs):
    """
    Bin stellar and planetary models onto a common wavelength grid.
    
    This is the core computation function using Numba for speed.
    
    Parameters
    ----------
    wv1 : float
        Minimum wavelength (um)
    wv2 : float
        Maximum wavelength (um)
    dw : float
        Wavelength bin width (um)
    starmodel_wv : ndarray, shape (snpt,)
        Stellar model wavelengths
    starmodel_flux : ndarray, shape (snpt,)
        Stellar model flux
    ld_coeff : ndarray, shape (snpt, 4)
        Limb darkening coefficients for stellar model
    planetmodel_wv : ndarray, shape (pnpt,)
        Planetary model wavelengths
    planetmodel_rprs : ndarray, shape (pnpt,)
        Planetary model Rp/R* values
    bin_starmodel_wv : ndarray, shape (bmax,)
        Output: binned stellar wavelengths (modified in-place)
    bin_starmodel_flux : ndarray, shape (bmax,)
        Output: binned stellar flux (modified in-place)
    bin_ld_coeff : ndarray, shape (bmax, 4)
        Output: binned limb darkening coefficients (modified in-place)
    bin_planetmodel_wv : ndarray, shape (bmax,)
        Output: binned planetary wavelengths (modified in-place)
    bin_planetmodel_rprs : ndarray, shape (bmax,)
        Output: binned planetary Rp/R* values (modified in-place)
    """
    
    snpt = len(starmodel_wv)
    pnpt = len(planetmodel_wv)
    bmax = len(bin_starmodel_wv)
    
    # Initialize output arrays
    bin_starmodel_wv[:] = 0.0
    bin_starmodel_flux[:] = 0.0
    bin_ld_coeff[:] = 0.0
    bin_planetmodel_wv[:] = 0.0
    bin_planetmodel_rprs[:] = 0.0
    
    # Allocate bin count array
    bin_count = np.zeros(bmax, dtype=np.int64)
    
    # ============================================================
    # Bin Stellar Model
    # ============================================================
    
    for i in range(snpt):
        s_wv = starmodel_wv[i]  # current wavelength
        b = int((s_wv - wv1) / dw)  # bin number (0-indexed)
        
        # Check if bin is valid
        if b >= 0 and b < bmax:
            bin_starmodel_wv[b] += s_wv
            bin_starmodel_flux[b] += starmodel_flux[i]
            for j in range(4):
                bin_ld_coeff[b, j] += ld_coeff[i, j]
            bin_count[b] += 1
    
    # Average the binned stellar model
    for b in range(bmax):
        if bin_count[b] > 0:
            bin_starmodel_wv[b] /= bin_count[b]
            bin_starmodel_flux[b] /= bin_count[b]
            for j in range(4):
                bin_ld_coeff[b, j] /= bin_count[b]
        else:
            # If bin is empty, use bin center as wavelength
            bin_starmodel_wv[b] = dw * (b + 0.5) + wv1
    
    # ============================================================
    # Bin Planetary Model
    # ============================================================
    
    # Reset bin_count
    bin_count[:] = 0
    
    for i in range(pnpt):
        s_wv = planetmodel_wv[i]  # current wavelength
        b = int((s_wv - wv1) / dw)  # bin number (0-indexed)
        
        # Check if bin is valid
        if b >= 0 and b < bmax:
            bin_planetmodel_wv[b] += s_wv
            bin_planetmodel_rprs[b] += planetmodel_rprs[i]
            bin_count[b] += 1
    
    # Average the binned planetary model
    for b in range(bmax):
        if bin_count[b] > 0:
            bin_planetmodel_wv[b] /= bin_count[b]
            bin_planetmodel_rprs[b] /= bin_count[b]
        else:
            # If bin is empty, use bin center as wavelength
            bin_planetmodel_wv[b] = dw * (b + 0.5) + wv1


def binmodels_py(wv1, wv2, dw, starmodel_wv, starmodel_flux, ld_coeff,
                 planetmodel_wv, planetmodel_rprs, bin_starmodel_wv,
                 bin_starmodel_flux, bin_ld_coeff, bin_planetmodel_wv,
                 bin_planetmodel_rprs):
    """
    Public wrapper function for binmodels_py that matches the original Fortran interface.
    
    Bins stellar and planetary model spectra onto a common wavelength grid.
    All output arrays are modified in-place.
    
    Parameters
    ----------
    wv1 : float
        Minimum wavelength (um)
    wv2 : float
        Maximum wavelength (um)
    dw : float
        Wavelength bin width (um)
    starmodel_wv : ndarray, shape (snpt,)
        Stellar model wavelengths
    starmodel_flux : ndarray, shape (snpt,)
        Stellar model flux
    ld_coeff : ndarray, shape (snpt, 4)
        Limb darkening coefficients for stellar model
    planetmodel_wv : ndarray, shape (pnpt,)
        Planetary model wavelengths
    planetmodel_rprs : ndarray, shape (pnpt,)
        Planetary model Rp/R* values
    bin_starmodel_wv : ndarray, shape (bmax,)
        Output: binned stellar wavelengths (modified in-place)
    bin_starmodel_flux : ndarray, shape (bmax,)
        Output: binned stellar flux (modified in-place)
    bin_ld_coeff : ndarray, shape (bmax, 4)
        Output: binned limb darkening coefficients (modified in-place)
    bin_planetmodel_wv : ndarray, shape (bmax,)
        Output: binned planetary wavelengths (modified in-place)
    bin_planetmodel_rprs : ndarray, shape (bmax,)
        Output: binned planetary Rp/R* values (modified in-place)
    
    Notes
    -----
    This is a Python translation of the original Fortran code (binmodels_py.f90)
    with Numba JIT compilation for performance.
    
    The algorithm:
    1. For each wavelength point in the input models, determine which bin it falls into
    2. Accumulate values in each bin
    3. Average the accumulated values by dividing by the count of points in each bin
    4. For empty bins, assign the bin center wavelength
    """
    
    # Ensure arrays are contiguous and have correct dtype
    starmodel_wv = np.asarray(starmodel_wv, dtype=np.float64, order='C')
    starmodel_flux = np.asarray(starmodel_flux, dtype=np.float64, order='C')
    ld_coeff = np.asarray(ld_coeff, dtype=np.float64, order='C')
    planetmodel_wv = np.asarray(planetmodel_wv, dtype=np.float64, order='C')
    planetmodel_rprs = np.asarray(planetmodel_rprs, dtype=np.float64, order='C')
    
    # Ensure output arrays are writable and contiguous
    bin_starmodel_wv = np.asarray(bin_starmodel_wv, dtype=np.float64, order='C')
    bin_starmodel_flux = np.asarray(bin_starmodel_flux, dtype=np.float64, order='C')
    bin_ld_coeff = np.asarray(bin_ld_coeff, dtype=np.float64, order='C')
    bin_planetmodel_wv = np.asarray(bin_planetmodel_wv, dtype=np.float64, order='C')
    bin_planetmodel_rprs = np.asarray(bin_planetmodel_rprs, dtype=np.float64, order='C')
    
    # Call the Numba-compiled kernel
    binmodels_py_kernel(wv1, wv2, dw, starmodel_wv, starmodel_flux, ld_coeff,
                        planetmodel_wv, planetmodel_rprs, bin_starmodel_wv,
                        bin_starmodel_flux, bin_ld_coeff, bin_planetmodel_wv,
                        bin_planetmodel_rprs)
