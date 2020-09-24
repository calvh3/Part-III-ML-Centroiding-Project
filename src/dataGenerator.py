import numpy as np
from numpy import sqrt
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
import MegaScreen #!pip install MegaScreen
import functools
import sys

def NumZernike(m):
    """Return the number of polynomials up to and including radial order m"""
    return (m + 1) * (m + 2) // 2

def jtonm(j):
    """Convert Noll index j (1-based) to radial index n and azimuthal index m.  
    The method is described by V. N. Mahajan, Chapter 13, 
    Optical Shop Testing, 3rd. Ed., D. Malacara. ed., Wiley,2007"""
    n = int(sqrt(2 * j - 1) + 0.5) - 1
    m = (1 - 2 * (j % 2)) * (
        2 * int((2 * j + 1 + (n % 2) - n * (n + 1)) / 4.0) - (n % 2)
    )
    return (n, m)

@functools.lru_cache()
def ZernikeGrid(gridSize, maxRadial, diameter=None, orthoganalise=True):
    """Return Zernike polynomials sampled on a square grid.
    Returns all the polynomials for radial orders up to and including maxRadial.
    If diameter is None, then the unit circle just touches the outer edges of the 
    square (with the convention that the square extends +-0.5 pixels beyond the grid).
    Uses a recursive method to evaluate the radial polynomial.
    Returns the polynomials in the order given by Noll (1976), but zero-based.
    """
    if diameter == None:
        diameter = gridSize
    radius = diameter / 2.0
    y, x = np.mgrid[0:gridSize, 0:gridSize]
    x = (x - (gridSize - 1.0) / 2.0) / radius
    y = (y - (gridSize - 1.0) / 2.0) / radius
    # Convert to polar co-ordinates
    temp = x + 1j * y
    r = np.abs(temp)
    R = KinterRadial(r, maxRadial)
    # Create entheta[n] = exp(i*n*theta) recursively from exp(i*theta)
    eitheta = np.where(r == 0.0, 1.0, temp / r)
    entheta = np.concatenate(
        (
            np.ones((1,) + eitheta.shape, dtype=np.complex),
            np.cumprod(np.broadcast_to(eitheta, (maxRadial,) + eitheta.shape), axis=0),
        )
    )
    cntheta = entheta.real
    sntheta = entheta.imag
    jmax = NumZernike(maxRadial)
    Z = np.empty((jmax,) + r.shape)
    for j in range(jmax):
        n, m = jtonm(j + 1)
        const = sqrt((2 - (m == 0)) * (n + 1))
        if m < 0:
            Z[j] = R[n, -m] * sntheta[-m] * const
        else:
            Z[j] = R[n, m] * cntheta[m] * const
    # Make zernike zero outside unit circle (useful for dot product)
    Z = Z * np.less_equal(r, 1.0)
    if orthoganalise:
        return Orthoganalise(Z)
    else:
        return Z

def KinterRadial(r, maxRadial):
    """Compute Zernike radial functions using the modified Kintner method.
    Discussion may be found in Chong, Chee-Way, P. Raveendran, and R. Mukundan. 
    'A Comparative Analysis of Algorithms for Fast Computation of Zernike Moments'.
    Pattern Recognition 36, no. 3 (March 2003): 731-42. 
    doi:10.1016/S0031-3203(02)00091-2."""
    # Storage array for result
    # Creates array 4 * larger than necessary, but simpler to code
    R = np.empty((maxRadial + 1, maxRadial + 1) + r.shape)
    # Calc diagonal elements recursively
    R[0, 0] = np.ones_like(r)
    q = np.arange(1, maxRadial + 1)
    R[q, q] = np.cumprod(np.broadcast_to(r, (maxRadial,) + r.shape), axis=0)
    # Calc second batch of elements according to Fig 3 in Chong et al.
    for q in range(maxRadial - 1):
        R[q + 2, q] = (q + 2) * R[q + 2, q + 2] - (q + 1) * R[q, q]
    # Calc remaining elements
    r2 = r ** 2
    for q in range(maxRadial - 3):
        for p in range(q + 4, maxRadial + 1, 2):
            k1 = (p + q) * (p - q) * (p - 2) / 2.0
            k2 = 2 * p * (p - 1) * (p - 2)
            k3 = -q * q * (p - 1) - p * (p - 1) * (p - 2)
            k4 = -p * (p + q - 2) * (p - q - 2) / 2.0
            R[p, q] = ((k2 / k1) * r2 + (k3 / k1)) * R[p - 2, q] + (k4 / k1) * R[
                p - 4, q
            ]
    return R

def Orthoganalise(Zernikes):
    """Orthoganalise Zernikes to mitigate effects of under-sampling.
    """
    colZernikes = np.transpose(np.reshape(Zernikes, (len(Zernikes), -1)))
    q, r  = np.linalg.qr(colZernikes, mode="reduced")
    orthoZernikes = np.reshape(np.transpose(q), Zernikes.shape)
    zernikeNorm = r[0,0]
    orthoZernikes *= zernikeNorm
    return orthoZernikes

def Phase_screen_generator(
    diameter, L0, maxRadial,r0,zernikeGrids):
    """Generate a simulated phase screen, return phase screen 
    and an array of its zernike polynomial coefficents"""
    #zernike = ZernikeGrid(diameter,maxRadial=maxRadial,diameter=None,orthoganalise=True)
    zernike = zernikeGrids
    result = []
    coefficients = np.zeros(len(zernike))
    for screen in MegaScreen.MegaScreen(
        r0=float(r0),
        L0=L0,
        windowShape=(diameter, diameter),
        numIter=1
    ):
        coefficients = zernike_coefficients(screen, zernike)
        result.append(coefficients)
    return np.array(result), screen

def zernike_coefficients(screen, zernike):
    """ Return Zernike coefficients for a given phase screen"""
    screen = screen * zernike[0]  # Cut out a circle
    num_radials = int(np.alen(zernike[:,0]))
    for i in range(num_radials):  #Normalisation
        zernike[i] = zernike[i]/np.sum(zernike[i]*zernike[i])        
    Z_coefficients = np.tensordot(zernike , screen)
    return Z_coefficients
    

def circular_mask(gridSize):
    """Returns a circular source point"""
    '''if mask_diameter == None:
        mask_diameter = gridSize'''
    mask_diameter= gridSize
    radius = mask_diameter / 2.0
    y, x = (np.mgrid[0:gridSize, 0:gridSize] - (gridSize - 1.0) / 2.0) / radius
    temp = x + 1j * y
    r = np.abs(temp)
    return np.less_equal(r, 1.0)

def speckle_image(screen, mask, output_diameter, oversample):
    """Creates speckle image by Fourier Transforming the source (circular_mask)
    with the phasescreen"""
    #Pupil is phase shift due to screen * circular source aperture
    if output_diameter == None:
        output_diameter = int(len(screen)*oversample)
    pupil = np.exp(1j * screen) * mask
    image = fftshift(abs(fft2(pupil, s=oversample * np.array(screen.shape)))) ** 2
    #Crop FFT plane into quarter screen
    width = len(image)
    min_w = int((width-output_diameter)/2)
    max_w = int(width-min_w)
    image = image[min_w:max_w,min_w:max_w]
    #image = image/np.sum(abs(image)) #Normalise image
    image = (image-np.abs(image).min())/(np.abs(image).max()-np.abs(image).min())
    '''In real algorithm need to normalise image using a threshold?'''
    return image

def centroid_position(SpeckleScreen):
    '''Returns the location of the image (COM) centroid for a 
       given SpeckleScreen, if normalise = True returns absolute 
       position (z_coefficients) otherwise returns centroid in pixels. '''
    diameter = len(SpeckleScreen)
    x = np.linspace(0.5,diameter-0.5,diameter)
    X_Int = np.sum(np.dot(SpeckleScreen,x)) 
    Y_Int = np.sum(np.dot(SpeckleScreen.T,x)) 
    I = np.sum(SpeckleScreen)
    x_pos = (X_Int/I - (diameter)/2-0.5) 
    y_pos = (Y_Int/I - (diameter)/2-0.5) 
    #x_pos = x_pos / normalisation
    #y_pos = y_pos / normalisation
    return x_pos, y_pos
    
def pre_process(SpeckleScreen, c_p, output_diameter):
    #Specklescreen_processed = shift(SpeckleScreen,(-c_p[1],-c_p[0]),mode='constant',cval=0)
    Specklescreen_processed = np.roll(SpeckleScreen,-int(c_p[0]),axis=1)
    Specklescreen_processed = np.roll(Specklescreen_processed,-int(c_p[1]),axis=0)
    width = len(Specklescreen_processed)
    min_w = int((width-output_diameter)/2)
    max_w = int(width-min_w)
    Specklescreen_processed = Specklescreen_processed[min_w:max_w,min_w:max_w]
    return Specklescreen_processed

def pre_process2(SpeckleScreen, output_diameter):
    width = len(SpeckleScreen)
    min_w = int((width-output_diameter)/2)
    max_w = int(width-min_w)
    SpeckleScreen = SpeckleScreen[min_w:max_w,min_w:max_w]
    return SpeckleScreen    


def downsampler(SpeckleScreen,k_size,batch_size=None):
    image_diameter = np.alen(SpeckleScreen[0,:,0])
    down_diameter = int(image_diameter/k_size)
    if batch_size == None:
      batch_size = len(SpeckleScreen)//5
    def downsample_batch(SpeckleScreen_d,k_size):
        '''downsamples a batch of images by summing over a k_size*k_size square kernel
        (Vectorised)'''
        o_res = len(SpeckleScreen_d[0,:,0])
        n_res = o_res // k_size
        SpeckleScreen_d = SpeckleScreen_d.reshape(len(SpeckleScreen_d),n_res,k_size,n_res,k_size).sum(axis=(2,4))
        return SpeckleScreen_d

    SpeckleScreen_final = np.zeros((len(SpeckleScreen),down_diameter,down_diameter))
    for i in range(int(len(SpeckleScreen)/batch_size)):
        SpeckleScreen_d = SpeckleScreen[i*batch_size:(i+1)*batch_size,0:image_diameter,0:image_diameter]
        SpeckleScreen_final[i*batch_size:(i+1)*batch_size,0:down_diameter,0:down_diameter] = downsample_batch(SpeckleScreen_d,k_size)
        #print('downsampled',batch_size*(i+1),'images')
    return SpeckleScreen_final/(k_size*k_size)

def downsample(SpeckleScreen,k_size):
    '''
    downsample a signle image
    '''
    def normalise(image):
        image1 = (image-np.abs(image).min())*1/(np.abs(image).max()-np.abs(image).min())
        return image1
    #downsamples image by summing over a k_size*k_size square kernel
    o_res = len(SpeckleScreen[:,0])
    n_res = o_res // k_size
    SpeckleScreen = SpeckleScreen[:n_res*k_size,:n_res*k_size].reshape(n_res,k_size,n_res,k_size).sum(axis=(1,3))
    SpeckleScreen = normalise(SpeckleScreen)
    return SpeckleScreen

def normalise(images):
    def normalise_image(image):
        image1 = (np.abs(image)-np.abs(image).min())*1/(np.abs(image).max()-np.abs(image).min())
        return image1
    images2 = np.zeros(np.shape(images))
    for i in range(np.alen(images)):
        image3 = images[i,:,:]
        image3 = normalise_image(image3)
        images2[i,:,:] = image3
    return images2

def noiser(SpeckleScreen, photon_count):
    power = np.sum(SpeckleScreen,axis=(1,2))
    print('power shape',np.shape(SpeckleScreen))
    SpeckleScreen = SpeckleScreen/power[:,np.newaxis,np.newaxis]*photon_count
    SpeckleScreen = np.random.poisson(np.abs(SpeckleScreen))
    SpeckleScreen = SpeckleScreen*power[:,np.newaxis,np.newaxis]/photon_count
    return SpeckleScreen

def pre_centroid(SpeckleScreen, c_p):
    #Specklescreen_processed = shift(SpeckleScreen,(-c_p[1],-c_p[0]),mode='constant',cval=0)
    Specklescreen_processed = np.roll(SpeckleScreen,-int(c_p[0]),axis=1)
    Specklescreen_processed = np.roll(Specklescreen_processed,-int(c_p[1]),axis=0)
    return Specklescreen_processed

def crop(SpeckleScreen, output_diameter):
    width = len(SpeckleScreen)
    min_w = int((width-output_diameter)/2)
    max_w = int(width-min_w)
    SpeckleScreen = SpeckleScreen[min_w:max_w,min_w:max_w]
    return SpeckleScreen    

def cropbatch(SpeckleScreen, output_diameter):
    width = len(SpeckleScreen[0,:,0])
    min_w = int((width-output_diameter)/2)
    max_w = int(width-min_w)
    SpeckleScreen = SpeckleScreen[:,min_w:max_w,min_w:max_w]
    return SpeckleScreen    


def normalise_centroid(diameter,oversample):
    '''Use to determine the normalisation parameter for 
       centroid_position to predict G-tilts'''
    zernike_test = ZernikeGrid(gridSize=diameter, maxRadial=1,orthoganalise=True)
    Phasescreen_test = zernike_test[0]*zernike_test[2]

    plt.figure(figsize=(7,7))
    plt.imshow(Phasescreen_test)
    plt.colorbar()
    plt.show()


    plt.figure(figsize=(7,7))
    plt.imshow(zernike_test[0]*zernike_test[1])
    plt.colorbar()
    plt.show()

    SpeckleScreen_test = speckle_image(Phasescreen_test,circular_mask(diameter),output_diameter=diameter*oversample,oversample=oversample)
    
    plt.figure(figsize=(7,7))
    plt.imshow(crop(SpeckleScreen_test,diameter//10))
    plt.show()

    normalisation = centroid_position(SpeckleScreen_test)
    print('y error is', normalisation[1])
    normalisation = normalisation[0]
    print('normalisation factor is', normalisation)
    return normalisation  

class generate:
  '''Generate class outlines how to make one data sample. Creates a random phasescreen, 
     fourier transforms into diffraction plane, and computes the tip/tilt zernike polynomials.'''
  def __init__(self,L0,r0,diameter,oversample,zernikeGrids):
      Zernikes, phasescreen  = Phase_screen_generator(diameter,L0, 1,r0,zernikeGrids)
      self.Zs = Zernikes[0,1:3]
      self.screen = speckle_image(phasescreen, circular_mask(diameter),output_diameter=diameter*oversample,oversample=oversample)
      self.oversampling = oversample

  def downsample(self,k_size):
        self.screen = downsample(self.screen,k_size)
  
  def centroid(self):
        c_p = centroid_position(self.screen)
        normalisation = self.oversampling*2.546198071489925/4
        self.Gs = np.divide(c_p,normalisation)
        self.screen = pre_centroid(self.screen, c_p)
  
  def crop(self,output_diameter):
      self.screen = crop(self.screen,output_diameter)
  
  def normalise(self):
      self.screen = (self.screen-np.abs(self.screen).min())*1/(np.abs(self.screen).max()-np.abs(self.screen).min())
    

'''Generate zernike screens once for runtime'''
zernikegrids = ZernikeGrid(diameter,maxRadial=1,diameter=None,orthoganalise=True)
zernikegrids[2,:,:] = -zernikegrids[2,:,:]
 
    

'''Example File Creation/Output:'''
    
from astropy.io import fits

def generate_batch(n,L0,r0,diameter,oversample,output_diameter):
    #Generate a batch of specklescreens, Zs, Gs
    specklescreens = np.zeros((n,output_diameter,output_diameter))
    Zs = np.zeros((n,2))
    Gs = np.zeros((n,2))

    for i in range(n):
        speckle = generate(L0,r0,diameter,oversample,zernikegrids)
        speckle.centroid()
        speckle.downsample(2)
        Gs[i,:] = speckle.Gs
        speckle.crop(output_diameter)
        specklescreens[i,:,:] = speckle.screen
        Zs[i,:] = speckle.Zs
        if not i%50:
            print('step:',i)
    return specklescreens, Zs, Gs

def FITSsave(image,path):
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul.append(fits.ImageHDU(data=image))
    hdul.writeto(path) 

    
for i in range(batches):
    specklescreens, Zs, Gs = generate_batch(n,L0,r0,diameter,oversample,output_diameter)
    print("batch",i+1,"complete")

    path1 = '/content1/specklescreens{0}.fits'.format(i)
    FITSsave(specklescreens,path1)

    path2 = '/content2/Zs{0}.fits'.format(i)
    FITSsave(Zs,path2)

    path3 = '/content3/Gs{0}.fits'.format(i)
    FITSsave(Gs,path3)
