import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage as sk
from scipy import ndimage as ndi

from skimage import exposure
from skimage import io
from skimage import feature
from skimage import filters
from skimage import morphology

# for HEQ
from skimage.filters import rank
from skimage.morphology import disk
from skimage.feature import peak_local_max


# convenience functions for printing histogram + img
def img_hist_single(image):

    f, ax = plt.subplots(ncols=2, figsize=(20,10));
    ax[0].hist(image.ravel(), bins=256)
    ax[1].imshow(image, cmap='gray')
    
def img_hist_stack(image, stack):

    f, ax = plt.subplots(ncols=2, figsize=(20,10));
    ax[0].hist(image[stack,:, :].ravel(), bins=256)
    ax[1].imshow(image[stack,:, :], cmap='gray')


# local histogram equalization
def local_eq(img, selem=disk(5000), **kwargs):
    """
    Equalize image using local (user-defined) histogram

    An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image. 
    Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.
    
    ------------
    Parameters:
    ------------
    mage : 2-D array (uint8, uint16)
        Input image.
    
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).  
                
    ------------  
    Returns:
    ------------  
    out: (M, N[,C]) ndarray
        output image
    
    
    ------------
    Notes:
    ------------
    - For color images, the following steps are performed:
        - The image is converted to HSV color space
        - The CLAHE algorithm is run on the V (Value) channel
        - The image is converted back to RGB space and returned
    
    
    ------------
    References:
    ------------
    [1]: http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    [2]: http://en.wikipedia.org/wiki/Histogram_equalization


    """
    

    img_local_eq = np.empty_like(img)
    for z in range(len(img)):
        img_local_eq[z, :, :] = rank.equalize(img[z, :, :], selem)

    return img_local_eq


# contrast-based intensity stretching

def contrast_stretch(img, p1=2, p2=98, **kwargs):
    """
    intensity-based contrast stretching/shrinking. 
    returns a z-stack with each plane stretched/shrunk.

    An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image. 
    Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.
    
    ------------
    Parameters:
    ------------
    image: (M, N[,C]) ndarray
        input image
        
    in_range, out_range: str or 2-tuple
        Min and max intesnity values of input and output image. 
        The possible values for this parameter are enumerated below.
    
        'image': use image min/max as the intensity range
        
        'dtype': use min/max of the images dtype as the intensity range
        
        'dtype-name': use intensity range absed on desired dtype. Must be valid key in DTYPE_RANGE
        
        '2-tuple': Use range_values as explicit min/max intensities
            
                
    ------------  
    Returns:
    ------------  
    out: (M, N[,C]) ndarray
        equalized image
    
    
    ------------
    Notes:
    ------------
    - For color images, the following steps are performed:
        - The image is converted to HSV color space
        - The CLAHE algorithm is run on the V (Value) channel
        - The image is converted back to RGB space and returned
    
    
    ------------
    References:
    ------------
    [1]: http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    [2]: http://en.wikipedia.org/wiki/Histogram_equalization


    """
    
    pct1, pct2 = np.percentile(img, (p1, p2))
    
    img_cs = np.empty_like(img)
    for z in range(len(img)):
        img_cs[z, :, :] = exposure.rescale_intensity(img[z, :, :], in_range=(pct1, pct2), **kwargs)
    return img_cs


# heq - histogram-based equalization

def heq(img, **kwargs):
    """
    takes an image containing z-stacks and applies a histogram equalization algorithm 
    to each plane of the stack and returns the equalized z-stack
    
    An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image. 
    Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.
    
    ------------
    Parameters:
    ------------
    image: (M, N[,C]) ndarray
        input image
        
    nbins: int, optional
        Number of gray bins for histogram ('data range')
     
     mask: ndarray of bools or 0s and 1s, optional
         Array of same shape as image. Only points at which mask==True are uesd for the equalization, 
         which is applied to the whole image.
    
    ------------  
    Returns:
    ------------  
    out: (M, N[,C]) ndarray
        equalized image
    
    
    ------------
    Notes:
    ------------
    - For color images, the following steps are performed:
        - The image is converted to HSV color space
        - The CLAHE algorithm is run on the V (Value) channel
        - The image is converted back to RGB space and returned
    
    ------------
    References:
    ------------
    [1]: http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    [2]: http://en.wikipedia.org/wiki/Histogram_equalization

    """
       
    img_heq = np.empty_like(img)
    for z in range(len(img)):
        img_heq[z, :, :] = exposure.equalize_hist(img[z, :, :], **kwargs)
    return img_heq


# Adaptive equalization - CLAHE

# adapt_eq - adaptive histogram equalization

def adapt_eq(img, clip_limit=0.03, **kwargs):
    """
    takes an image containing z-stacks and applies an adaptive histogram equalization algorithm 
    to each plane of the stack and returns the equalized z-stack
    
    Utilizes CLHAE: Contrast Limited Adaptive Histogram Equalization.

    An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image. 
    Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.
    
    ------------
    Parameters:
    ------------
    image: (M, N[,C]) ndarray
        input image
    
    kernel_size: integer or list-like, optional
        Defines the shape of contextual regions used in the algorithm. 
        If iterable is passed, it must have the same number of elements as image.ndim (without color channel). 
        If integer, it is broadcasted to each image dimension. 
        By default, kernel_size is 1/8 of image height by 1/8 of its width.
    
    clip_limit: float_optional
        Clipping limit, normalized between 0 and 1 (higher values give more contrast)
        
    nbins: int, optional
        Number of gray bins for histogram ('data range')
        
    
    ------------  
    Returns:
    ------------  
    out: (M, N[,C]) ndarray
        equalized image
    
    
    ------------
    See also:
    ------------
    equalize_hist, rescale_intensity
    
    ------------
    Notes:
    ------------
    - For color images, the following steps are performed:
        - The image is converted to HSV color space
        - The CLAHE algorithm is run on the V (Value) channel
        - The image is converted back to RGB space and returned
    
    ------------
    References:
    ------------
    [1]: http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
    [2]: http://tog.acm.org/resources/GraphicsGems/
    [3]: https://en.wikipedia.org/wiki/CLAHE#CLAHE

    """
       
    img_adeq = np.empty_like(img)
    for z in range(len(img)):
        img_adeq[z, :, :] = exposure.equalize_adapthist(img[z, :, :], **kwargs)
    return img_adeq
