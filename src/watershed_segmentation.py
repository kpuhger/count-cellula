import skimage as sk
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

from skimage import io, feature, filters, color, restoration, util
from skimage.morphology import watershed
from skimage.future import graph
from skimage.filters import threshold_otsu, threshold_local
from scipy import ndimage as ndi


exec(open('/Users/kpuhger/Documents/count-cellula/src/heq.py').read())
exec(open('/Users/kpuhger/Documents/count-cellula/src/smoothing.py').read())



# step 1 gaussian blur image
img_gauss = gaussian_blur(img, 3)

# step 2 perform adaptive equalization on blurred image
img_gauss_aeq = adapt_eq(img_gauss)


# get edges and peaks and perform watershed segmentation
def edges_stack(img, **kwargs):
    # get edges
    edges = np.empty_like(img)
    for z in range(len(img)):
        edges[z] = filters.sobel(img[z])
    return edges

    # get peaks

def peak_stack(img, **kwargs):
    edges = edges_stack(img)
    threshold = np.empty_like(img)
    # need to figure this out
    non_edges = np.empty_like(img)
    distance_from_edge = np.empty_like(img)
#     peaks_image = np.empty_like(img)
    seeds = np.empty_like(img)
    
    for z in range(len(img)):
        threshold[z, :, :] = filters.threshold_otsu(edges[z, :, :])
        non_edges[z, :, :] = (edges[z, :, :] < threshold[z, :, :])
        distance_from_edge[z, :, :] = ndi.distance_transform_edt(non_edges[z, :, :])

    # return threshold, non_edges, distance_from_edge
        
    for z in range(len(img)):
        if z == 0:
            peaks = feature.peak_local_max(distance_from_edge[0, :, :], min_distance=9)
#             peaks = np.column_stack((peaks, np.zeros((len(peaks)))))
#             peaks_image = np.zeros(img[0, :, :].shape, np.bool)
#             peaks_image[0, :, :][tuple(np.transpose(peaks[0, :, :]))] = True
#             seeds[0, :, :], num_seeds = ndi.label(peaks_image[0, :, :])
#             peaks_image[z, :, :] = np.zeros(img[z, :, :].shape, np.bool)
#             peaks_image_z = peaks_image[0, :, :]
            peaks_image = np.zeros(img[0, :, :].shape, np.bool)
            peaks_image[peaks_image[tuple(np.transpose(peaks))]] = True
            seeds[0, :, :], num_seeds = ndi.label(peaks_image)
            peaks = np.column_stack((peaks, np.zeros((len(peaks)))))
        else:
            new_peaks = feature.peak_local_max(distance_from_edge[z, :, :], min_distance=9)
            new_peaks[new_peaks[tuple(np.transpose(new_peaks))]]= True
#             new_peaks_image = np.zeros(img[z, :, :].shape, np.bool)
#             new_peaks_image[z, :, :][tuple(np.transpose(peaks[z, :, :]))] = True
#             new_seeds[z, :, :], num_seeds = ndi.label(peaks_image[z, :, :]) 
            new_peaks = np.column_stack((new_peaks, np.full((len(new_peaks)), z)))
            peaks = np.concatenate((peaks, new_peaks))

#         peaks[z, :, :] = feature.peak_local_max(distance_from_edge[z, :, :], min_distance=9)
            peaks_image = np.zeros(img[z, :, :].shape, np.bool)
            peaks_image[peaks_image[tuple(np.transpose(peaks))]] = True
            seeds[z, :, :], num_seeds = ndi.label(peaks_image[z, :, :])
    
    return seeds

# watershed
def ws_stack(img, **kwargs):
    ws = np.empty_like(img)
    for z in range(len(img)):
        ws[z, :, :] = watershed(edges[z, :, :], seeds[z, :, :])
    return ws




