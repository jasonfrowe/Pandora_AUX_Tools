import numpy as np #numpy gives us better array management
import binmodels_py as bm #Python/Numba routine for speedy resampling of data

#Used to read in Kernels
from scipy.io import loadmat

#PSF convolution
#from scipy.signal import fftconvolve

from scipy import interpolate #spline interpolation

#Used to resize Kernels
from skimage.transform import downscale_local_mean, resize
from skimage import transform as tf #To manipulate images

from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.notebook import trange

# New pytfit5 transit model framework
import pytfit5.transitmodel as transitm
import pytfit5.keplerian as kep

### Functions that need to be updated for Pandora

class ModelPars:
    """Default Model Parameters
    """

    nplanetmax=9 #code is hardwired to have upto 9 transiting planets.
    #default parameters -- these will cause the program to end quickly
    tstart=0.0 #start time (days)
    tend=1.0 #end time (days)
    iframe=30 #number of frames used to generate exposure time (needed for pointing jitter)
    exptime=30.0 #exposure time (s)
    deadtime=0.0 #dead time (s)
    modelfile='null' #stellar spectrum file name
    nmodeltype=2 #stellar spectrum type. 1=BT-Settl, 2=Atlas-9+NL limbdarkening
    rvstar=0.0 #radial velocity of star (km/s)
    vsini=0.0 #projected rotation of star (km/s)
    pmodelfile=[None]*nplanetmax #file with Rp/Rs values
    pmodeltype=[None]*nplanetmax #Type of planet file
    emisfile=[None]*nplanetmax #file with emission spectrum
    ttvfile=[None]*nplanetmax #file with O-C measurements
    #nplanet is tracked by pmodelfile.
    nplanet=0 #number of planets -- default is no planets - you will get staronly sim.
    sol=np.zeros(nplanetmax*8+1)
    sol[0]=1.0 #mean stellar density [g/cc]
    xout=2048  #dispersion axis
    xpad=10    #padding to deal with convolution fall-off
    ypad=10    #padding to deal with convolution fall-off
    yout=256   #spatial axis
    noversample=2 #oversampling
    gain=1.6 # electronic gain [e-/adu]
    saturation=65536.0 #saturation
    jitter_dis=1.27 #pointing jitter in dispersion axis [pixels, rms]
    jitter_spa=0.34 #pointing jitter in spatial axis [pixels, rms]
    
## Functions for scene-generation

def p2w(p,noversample,ntrace):
    """Usage: w=p2w(p,noversample,ntrace) Converts x-pixel (p) to wavelength (w)
    Inputs:
      p : pixel value along dispersion axis (float) on oversampled grid.
      noversample : oversampling factor (integer 10 >= 1)
      ntrace : order n=1,2,3

    Outputs:
      w : wavelength (um)
    """

    #co-efficients for polynomial that define the trace-position
    nc=5 #number of co-efficients
    #c=[[2.60188,-0.000984839,3.09333e-08,-4.19166e-11,1.66371e-14],\
    # [1.30816,-0.000480837,-5.21539e-09,8.11258e-12,5.77072e-16],\
    # [0.880545,-0.000311876,8.17443e-11,0.0,0.0]]
    #c=[[1.7,-3.63636e-3,0.0,0.0,0.0]]
    c=[[0.900556,0.00262894,1.06383e-05,2.57147e-08,-1.44608e-10]]
    
    pix=p/noversample
    w=c[ntrace-1][0]
    for i in range(1,nc):
        #print(w)
        w+=np.power(pix,i)*c[ntrace-1][i]
    #w*=10000.0 #um to A

    return w

def w2p(w,noversample,ntrace):
    """Usage: p=w2p(w,noversample,ntrace) Converts wavelength (w) to x-pixel (p)
    Inputs:
      w : wavelength (um)
      noversample : oversampling factor (integer 10 >= 1)
      ntrace : order n=1,2,3

    Outputs:
      p : pixel value along dispersion axis (float) on oversampled grid.

    """

    nc=5

    #c=[[2957.38,-1678.19,526.903,-183.545,23.4633],\
    #   [3040.35,-2891.28,682.155,-189.996,0.0],\
    #   [2825.46,-3211.14,2.69446,0.0,0.0]]
    #c=[[4.675e2,-2.75e2,0.0,0.0,0.0]]
    c=[[-948.718,2200.26,-1871.25,772.627,-120.033]]
    
    #wum=w/10000.0 #A->um
    p=c[ntrace-1][0]
    for i in range(1,nc):
        #print(p)
        p+=np.power(w,i)*c[ntrace-1][i]
    p=p*noversample

    return p

def ptrace(px,noversample,ntrace):
    """given x-pixel, return y-position based on trace
    Usage:
    py = ptrace(px,noversample,ntrace)

    Inputs:
      px : pixel on dispersion axis (float) on oversampled grid.
      noversample : oversampling factor (integer 10 >= 1)
      ntrace : order n=1,2,3

    Outputs:
      py : pixel on spatial axis (float) on oversampled grid.
    """
    nc=5 #number of co-efficients
    #c=[[275.685,0.0587943,-0.000109117,1.06605e-7,-3.87e-11],\
    #  [254.109,-0.00121072,-1.84106e-05,4.81603e-09,-2.14646e-11],\
    #  [203.104,-0.0483124,-4.79001e-05,0.0,0.0]]
    c=[[64.0,0.0,0.0,0.0,0.0]]
    
    
    opx=px/noversample #account for oversampling

    ptrace=c[ntrace-1][0]
    for i in range(1,nc):
        #print(w)
        ptrace+=np.power(opx,i)*c[ntrace-1][i]

    ptrace=(ptrace)*noversample
    return ptrace;

def readkernels(psf_dir,psf_names, pars):
    """Reads in PSFs from Matlab file and resamples to match pixel grid of simulation.
    
    Usage: psf=readkernels(psf_dir,psf_names)
    
        Inputs:
            psf_dir - location of PSFs
            psf_names - names of the PSF files to read in.  Order should match 'psf_wv' array
          
        Outputs:
            psf - array of PSFs.
    """
    
    detector_pixscale=18 #detector pixel size (microns)  ***This should be a model parameter***
    
    psf=[]
    for name in psf_names:
        mat_dict=loadmat(psf_dir+psf_names[0]) #read in PSF from Matlab file
        
        psf_native=mat_dict['psf']
        dx_scale=mat_dict['dx'] #scale in micron/pixel of the PSF
        x_scale=int(psf_native.shape[0]*dx_scale/detector_pixscale*pars.noversample) #This gives the PSF size in pixels 
        y_scale=int(psf_native.shape[1]*dx_scale/detector_pixscale*pars.noversample) #This gives the PSF size in pixels 
        
        #We now resize the PSF from psf.shape to x_scale,yscale
        psf_resize=resize(psf_native,(x_scale,y_scale))
        psf.append(psf_resize)
        
        #plt.imshow(psf_resize,norm=LogNorm())
        #plt.show()
        
    return psf

def get_qe_from_wavelength(lam):
    """
    from the JWST NIRCam models
    """
    sw_coeffs = np.array([0.65830, -0.05668, 0.25580, -0.08350])
    sw_exponential = 100.0
    sw_wavecut = 2.38
    sw_qe = (
        sw_coeffs[0]
        + sw_coeffs[1] * lam
        + sw_coeffs[2] * lam ** 2
        + sw_coeffs[3] * lam ** 3
    )
    # if lam > sw_wavecut:
    #     sw_qe = sw_qe * np.exp((sw_wavecut - lam) * sw_exponential)
    return sw_qe

### List of Functions -- these will be made external ###

class specdata_class:
    def __init__(self):
        self.time=[]  #initialize arrays
        self.data=[]
        self.exptime=[]

class const_class:
    def __init__(self):
        self.Rearth=6.3781e6 #Radius of the Earth (m)
        self.Rsun=695508*1000 #Radius of the Sun (m) 


class response_class:
    def __init__(self):
        self.wv=[]  #initialize arrays
        self.response=[]
        self.quantum_yield=[]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
        
def readstarmodel(starmodel_file, nmodeltype=2, quiet=False):
    """Usage: starmodel_wv,starmodel_flux=readstarmodel(starmodel_file,smodeltype)
    Inputs:
      starmodel_file - full path and filename to star spectral model
      nmodeltype - type of model.  2==ATLAS

    Returns:
      starmodel_wv : wavelength (A)
      starmodel_flux : flux
      ld_coeff : non-linear limb-darkening coefficients
    """
    if not quiet: print('Reading star model atmosphere {:}'.format(starmodel_file))

    starmodel_wv=[]
    starmodel_flux=[]
    ld_coeff=[]

    if nmodeltype==2:

        f = open(starmodel_file,'r')
        for line in f:
            line = line.strip() #get rid of \n at the end of the line
            columns = line.split() #break into columns with space delimiter
            starmodel_wv.append(float(columns[0])/10000)
            flux=-float(columns[5])*np.pi*(42.0*float(columns[1])+70.0*float(columns[2])\
                    +90.0*float(columns[3])+105.0*float(columns[4])-210.0)/210.0
            starmodel_flux.append(np.max([0.0,flux]))
            ld_coeff.append([float(columns[1]),float(columns[2]),float(columns[3])\
                ,float(columns[4])])
        f.close()

        starmodel_wv=np.array(starmodel_wv)
        starmodel_flux=np.array(starmodel_flux)
        ld_coeff=np.array(ld_coeff)

    else:
        print('Currently on ATLAS-9 models are supported (nmodeltype=2)')

    return starmodel_wv,starmodel_flux,ld_coeff;

def readplanetmodel(planetmodel_file, planet_radius, star_radius, const, quiet=False):
    """Usage: planetmodel_wv,planetmodel_depth=readplanetmodel(planetmodel_file,pmodeltype)
    Inputs
      planetmodel_file : full path to planet model (wavelength,contrast atmosphere height)
        -- Generated using PSG with Contrast (transit atmospheric thickness, km)
      planet_radius : Planet Radius relative to the Rearth

    Outputs
      planetmodel_wv : array with model wavelengths (um)
      planetmodel_rprs : array with model r/R* values.
    """

    if not quiet: print('Reading planet atmosphere model {:}'.format(planetmodel_file))

    rp=planet_radius*const.Rearth #planet radius in m
    rs=star_radius*const.Rsun #star radius in m
        
    planetmodel_wv=[]
    planetmodel_rprs=[]
    f = open(planetmodel_file,'r')
    for line in f:
        line = line.strip() #get rid of \n at the end of the line
        columns = line.split() #break into columns with comma
        if is_number(columns[0]): #ignore lines that start with '#'
            planetmodel_wv.append(float(columns[0])) #wavelength (um)
            planetmodel_rprs.append((1000*float(columns[1])+rp)/rs) #r/R*
    f.close()

    planetmodel_wv=np.array(planetmodel_wv)       #convert to numpy array
    planetmodel_rprs=np.array(planetmodel_rprs)

    return planetmodel_wv,planetmodel_rprs

def get_dw(starmodel_wv,planetmodel_wv,pars):
    """ Get optimum wavelength grid.
    Usage: dw,dwflag=get_dw(starmodel_wv,planetmodel_wv,pars)

    Inputs
      starmodel_wv - star model wavelength array
      planetmodel_wv - planet model wavelength array
      pars - model parameters

    Outputs
      dw - optimum wavelength spacing
      dwflag - flag : 0-grid is good.  1-grid is too course

    """

    norder=1 #Order to use.

    #get spectral resolution of star spectra
    nstarmodel=len(starmodel_wv)
    dw_star_array=np.zeros(nstarmodel-1)
    sortidx=np.argsort(starmodel_wv)
    for i in range(nstarmodel-1):
        dw_star_array[i]=starmodel_wv[sortidx[i+1]]-starmodel_wv[sortidx[i]]
    dw_star=np.max(dw_star_array)
    #print('dw_star',dw_star)

    #get spectral resolution of planet spectra
    nplanetmodel=len(planetmodel_wv)
    print(nplanetmodel)
    dw_planet_array=np.zeros(nplanetmodel-1)
    sortidx=np.argsort(planetmodel_wv)
    for i in range(nplanetmodel-1):
        dw_planet_array[i]=planetmodel_wv[sortidx[i+1]]-planetmodel_wv[sortidx[i]]
    dw_planet=np.max(dw_planet_array)
    #print('dw_planet',dw_planet)

    #get spectra resolution needed to populate grid.
    xmax=pars.xout*pars.noversample
    dw_grid_array=np.zeros(xmax-1)
    for i in range(xmax-1):
        dw_grid_array[i]=p2w(i+1,pars.noversample,norder)\
          -p2w(i,pars.noversample,norder)
    dw_grid=np.abs(np.min(dw_grid_array))
    #print('dw_grid',dw_grid)

    dw=np.max((dw_star,dw_planet))

    if dw>dw_grid:
        print("Warning. stellar/planet model spectral resolution is too low.  Data will be interpolated.")
        dw=np.min((dw,dw_grid))
        dwflag=1
    else: #bin data to common grid.
        dw
        dwflag=0

    #print('dw',dw)

    return dw,dwflag

def resample_models(dw,starmodel_wv,starmodel_flux,ld_coeff,\
    planetmodel_wv,planetmodel_rprs,pars):
    """Resamples star and planet model onto common grid.

    Usage:
    bin_starmodel_wv,bin_starmodel_flux,bin_ld_coeff,bin_planetmodel_wv,bin_planetmodel_rprs\
      =resample_models(dw,starmodel_wv,starmodel_flux,ld_coeff,\
      planetmodel_wv,planetmodel_rprs,pars)


      Inputs:
        dw - wavelength spacing.  This should be calculated using get_dw
        starmodel_wv - stellar model wavelength array
        starmodel_flux - stellar model flux array
        ld_coeff - non-linear limb-darkening coefficients array
        planetmodel_wv - planet model wavelength array
        planetmodel_rprs - planet model Rp/R* array
        pars - model parameters
        
      Output:
        bin_starmodel_wv - binned star wavelength array
        bin_starmodel_flux - binned star model array
        bin_ld_coeff - binned limb-darkening array
        bin_planetmodel_wv - binned planet wavelength array (should be same size as bin_starmodel_wv)
        bin_planetmodel_rprs - binned Rp/R* array
    """

    xpad = pars.xpad * pars.noversample
    wv1 = np.min([p2w(pars.xout + xpad+1,1,1), p2w(0 - xpad, 1, 1)])
    wv2 = np.max([p2w(pars.xout + xpad+1,1,1), p2w(0 - xpad, 1, 1)])
    #print('wv:',wv1,wv2)
    wv1 = 0.5
    wv2 = 3.0
    #wv1 = np.max((wv1, 0.5)) #make sure range is sane
    #wv2 = np.min((wv2, 3.0))

    #This is when multiple orders are present (slitless)
    #for norder in range(1,4):
    #    wv1=np.min((p2w(pars.xout+1,1,norder),wv1))
    #    wv2=np.max((p2w(0,1,norder),wv2))

    bmax = int((wv2 - wv1) / dw)

    snpt = starmodel_wv.shape[0]
    pnpt = planetmodel_wv.shape[0]

    ld_coeff_in = np.zeros((snpt, 4))

    bin_starmodel_wv = np.zeros(bmax)
    bin_starmodel_flux = np.zeros(bmax)
    bin_ld_coeff = np.zeros((bmax,4), order='F')
    bin_planetmodel_wv = np.zeros(bmax)
    bin_planetmodel_rprs = np.zeros(bmax)

    bm.binmodels_py(wv1, wv2, dw,\
        starmodel_wv, starmodel_flux, ld_coeff,\
        planetmodel_wv, planetmodel_rprs,\
        bin_starmodel_wv, bin_starmodel_flux, bin_ld_coeff, bin_planetmodel_wv, bin_planetmodel_rprs)

    # Make sure the array is sorted in increasing wavelengths
    ind = np.argsort(bin_starmodel_wv)
    bin_starmodel_wv = bin_starmodel_wv[ind]
    bin_starmodel_flux = bin_starmodel_flux[ind]
    bin_planetmodel_wv = bin_starmodel_wv[ind]
    bin_planetmodel_rprs = bin_planetmodel_rprs[ind]
    bin_ld_coeff = bin_ld_coeff[ind]

    return bin_starmodel_wv, bin_starmodel_flux, bin_ld_coeff, bin_planetmodel_wv, bin_planetmodel_rprs

def transitmodel (sol,time,ld1,ld2,ld3,ld4,rdr,tarray,\
    itime=-1, ntt=0, tobs=0, omc=0, dtype=0 ):
    """Get a transit model
    Usage: tmodel = transitmodel(sol,time,ld1,ld2,ld3,ld4,rdr,tarray,\
      itime=-1, ntt=0, tobs=0, omc=0, dtype=0)

    Inputs:
      sol - Transit model object (pytfit5.transitmodel.transit_model_class)
      time - time array (nwavelength,)
      ld1,ld2,ld3,ld4 - limb-darkening arrays (nwavelength,)
      rdr - Rp/R* array (nplanet, nwavelength)
      tarray - Thermal eclipse array (nplanet, nwavelength)
      itime - integration time, scalar or array (nwavelength,)

    Outputs:
      tmodel - relative flux at times given by time array (nwavelength,)
    """
    # New pytfit5 framework - sol is now a transit_model_class object
    # The model must be called for each wavelength separately since
    # rdr and ted vary with wavelength
    
    nwv = ld1.shape[0]
    tmodel = np.ones(nwv)
    
    # Extract planet index (assuming single planet)
    if len(rdr.shape) > 1:
        rdr_wv = rdr[0]  # Shape: (nwavelength,)
    else:
        rdr_wv = rdr
        
    if len(tarray.shape) > 1:
        ted_wv = tarray[0]  # Shape: (nwavelength,)
    else:
        ted_wv = tarray
    
    # Handle itime - convert to array if scalar
    if isinstance(itime, (int, float, np.integer, np.floating)):
        itime_arr = np.full(nwv, itime)
    else:
        itime_arr = itime
    
    # Loop over wavelengths and compute transit model for each
    for i in range(nwv):
        # Set wavelength-dependent limb-darkening coefficients
        sol.nl1 = ld1[i]
        sol.nl2 = ld2[i]
        sol.nl3 = ld3[i]
        sol.nl4 = ld4[i]
        
        # Set wavelength-dependent Rp/R* and thermal eclipse
        sol.rdr = [rdr_wv[i]]  # List with one element per planet
        sol.ted = [ted_wv[i]]  # List with one element per planet
        
        # Generate transit model for this wavelength
        # Pass time and itime as arrays with single element
        tmodel[i] = transitm.transitModel(sol, np.array([time[i]]), np.array([itime_arr[i]]))[0]
    
    return tmodel

def addflux2pix(px,py,pixels,fmod):
    """Usage: pixels=addflux2pix(px,py,pixels,fmod)

    Drizel Flux onto Pixels using a square PSF of pixel size unity
    px,py are the pixel position (integers)
    fmod is the flux calculated for (px,py) pixel
        and it has the same length as px and py
    pixels is the image.
    """

    xmax = pixels.shape[0] #Size of pixel array
    ymax = pixels.shape[1]

    pxmh = px-0.5 #location of reference corner of PSF square
    pymh = py-0.5

    dx = np.floor(px+0.5)-pxmh
    dy = np.floor(py+0.5)-pymh

    # Supposing right-left as x axis and up-down as y axis:
    # Lower left pixel
    npx = int(pxmh) #Numpy arrays start at zero
    npy = int(pymh)

    #print('n',npx,npy)
    
    #if (npx >= 0) & (npx < xmax) & (npy >= 0) & (npy < ymax) :
    #    pixels[npx,npy]=pixels[npx,npy]+fmod
    
    if (npx >= 0) & (npx < xmax) & (npy >= 0) & (npy < ymax) :
        pixels[npx,npy]=pixels[npx,npy]+fmod*dx*dy

    #Same operations are done for the 3 pixels other neighbouring pixels

    # Lower right pixel
    npx = int(pxmh)+1 #Numpy arrays start at zero
    npy = int(pymh)
    if (npx >= 0) & (npx < xmax) & (npy >= 0) & (npy < ymax) :
        pixels[npx,npy]=pixels[npx,npy]+fmod*(1.0-dx)*dy

    # Upper left pixel
    npx = int(pxmh) #Numpy arrays start at zero
    npy = int(pymh)+1
    if (npx >= 0) & (npx < xmax) & (npy >= 0) & (npy < ymax) :
        pixels[npx,npy]=pixels[npx,npy]+fmod*dx*(1.0-dy)

    # Upper right pixel
    npx = int(pxmh)+1 #Numpy arrays start at zero
    npy = int(pymh)+1
    if (npx >= 0) & (npx < xmax) & (npy >= 0) & (npy < ymax) :
        pixels[npx,npy]=pixels[npx,npy]+fmod*(1.0-dx)*(1.0-dy)
    
    return pixels;

def gen_unconv_image(pars,response,bin_starmodel_wv,bin_starmodel_flux,bin_ld_coeff,\
    bin_planetmodel_rprs,time,itime,sol,norder,xcen,ycen):

    xpad=pars.xpad*pars.noversample
    ypad=pars.ypad*pars.noversample
    #array to hold synthetic image
    xmax=pars.xout*pars.noversample+xpad*2
    ymax=pars.yout*pars.noversample+ypad*2

    pixels=np.zeros((xmax,ymax))

    #interpolate over response and quantum yield
    response_spl = interpolate.splrep(response.wv, response.response[norder-1], s=0)
    quantum_yield_spl  = interpolate.splrep(response.wv, response.quantum_yield, s=0)
    rmax=np.max(response.wv)
    rmin=np.min(response.wv)

    #Generate Transit Model
    npt=len(bin_starmodel_wv) #number of wavelengths in model
    time_array=np.ones(npt)*time   #Transit model expects array
    itime_array=np.ones(npt)*itime #Transit model expects array
    rdr_array=np.ones((1,npt))*bin_planetmodel_rprs #r/R* -- can be multi-planet
    tedarray=np.zeros((1,npt)) #secondary eclipse -- can be multi-planet
    planet_flux_ratio=transitmodel (sol, time_array,\
                            bin_ld_coeff[:,0], bin_ld_coeff[:,1], bin_ld_coeff[:,2],bin_ld_coeff[:,3],\
                            rdr_array,tedarray,itime=itime_array)
    
    xjit=np.random.normal()*pars.jitter_dis*pars.noversample
    yjit=np.random.normal()*pars.jitter_spa*pars.noversample
    #print(xjit,yjit)
    
    for k in range(bin_starmodel_wv.shape[0]):

        w=bin_starmodel_wv[k]
        i=w2p(w,pars.noversample,norder)+xpad+xjit
        j=ptrace(i,pars.noversample,norder)+ypad+yjit

        if (i<=xmax+1) & (i>=0) & (j<=ymax+1) & (j>=0): #check if pixel is on grid

            if w < rmax and w > rmin:
                response_one = interpolate.splev(w, response_spl, der=0)
                quantum_yield_one = interpolate.splev(w, quantum_yield_spl, der=0)
            else:
                response_one = 0
                quantum_yield_one = 0
            flux=planet_flux_ratio[k]*bin_starmodel_flux[k]*response_one*quantum_yield_one
            pixels=addflux2pix(i+xcen,j+ycen,pixels,flux)
        
    return pixels

def filter(x,f_cen,f_wid,k=0.5):
    
    ans =0.5+1/np.pi*np.arctan(k*(x-f_cen+f_wid))
    ans2=0.5-1/np.pi*np.arctan(k*(x-f_cen-f_wid))
    
    f=ans*ans2
    
    return f;

def gen_unconv_image_v2(pars,response,bin_starmodel_wv,bin_starmodel_flux,bin_ld_coeff,\
    bin_planetmodel_rprs,time,itime,sol,norder, xcen, ycen, star_flux_ratio, make_tmodel=0, pixels=None):

    xpad=pars.xpad*pars.noversample
    ypad=pars.ypad*pars.noversample
    #array to hold synthetic image
    xmax=pars.xout*pars.noversample+xpad*2
    ymax=pars.yout*pars.noversample+ypad*2

    if pixels is None:
        pixels=np.zeros((xmax,ymax))

    #interpolate over response and quantum yield
    response_spl = interpolate.splrep(response.wv, response.response[norder-1], s=0)
    quantum_yield_spl  = interpolate.splrep(response.wv, response.quantum_yield, s=0)
    rmax=np.max(response.wv)
    rmin=np.min(response.wv)

    #Generate Transit Model
    npt=len(bin_starmodel_wv) #number of wavelengths in model
    time_array=np.ones(npt)*time   #Transit model expects array
    rdr_array=np.ones((1,npt))*bin_planetmodel_rprs #r/R* -- can be multi-planet
    tedarray=np.zeros((1,npt)) #secondary eclipse -- can be multi-planet
    
    exptime1=(pars.exptime/pars.iframe)/86400
    time1=time-pars.exptime/(86400*2)
    itime_array=np.ones(npt)*exptime1 #Transit model expects array
    time_array=np.ones(npt)*time1
    
    eff_exptime=0
    
    #ic=0
    while eff_exptime<pars.exptime/86400:
        
        if make_tmodel == 0:
            planet_flux_ratio=transitmodel (sol, time_array,\
                bin_ld_coeff[:,0], bin_ld_coeff[:,1], bin_ld_coeff[:,2],bin_ld_coeff[:,3],\
                rdr_array,tedarray,itime=itime_array)
        else:
            planet_flux_ratio = np.ones(bin_starmodel_wv.shape[0])
    
        if pars.iframe > 1:
            xjit=np.random.normal()*pars.jitter_dis*pars.noversample
            yjit=np.random.normal()*pars.jitter_spa*pars.noversample
        else:
            xjit = 0.0
            yjit = 0.0
        #print(xjit,yjit)

        for k in range(bin_starmodel_wv.shape[0]):

            w=bin_starmodel_wv[k]
            i=w2p(w,pars.noversample,norder)+xpad+xjit
            j=ptrace(i,pars.noversample,norder)+ypad+yjit

            if (i<=xmax+1) & (i>=0) & (j<=ymax+1) & (j>=0): #check if pixel is on grid

                if w < rmax and w > rmin:
                    response_one = interpolate.splev(w, response_spl, der=0)
                    quantum_yield_one = interpolate.splev(w, quantum_yield_spl, der=0)
                else:
                    response_one = 0
                    quantum_yield_one = 0
                flux=planet_flux_ratio[k]*bin_starmodel_flux[k]*response_one*quantum_yield_one*star_flux_ratio
                pixels=addflux2pix(i+xcen,j+ycen,pixels,flux)
                
        #ic+=1
        #print(ic,eff_exptime*86400,time_array)
                
        eff_exptime+=exptime1
        time_array+=exptime1 #Update current time
        

    return pixels