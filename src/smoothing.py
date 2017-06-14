# Mean blur. Increased disk_size = more smoothing
def mean_blur(image, disk_size):
    img_meanfilt = np.empty_like(image)
    for i in range(len(image)):
        img_meanfilt[i,:,:] = filters.rank.mean(image[i,:,:], disk(disk_size))
    return img_meanfilt

# Median blur. Increased disk_size = more smoothing
def median_blur(image, disk_size):
    img_medianfilt = np.empty_like(image)
    for i in range(len(image)):
        img_medianfilt[i,:,:] = filters.rank.median(image[i,:,:], disk(disk_size))
    return img_medianfilt

# Gaussian blur. Increased sigma = more smoothing
def gaussian_blur(image, sigma):
    # must convert image to float for this to work in a loop (unknown reason)
    image = (image*1.0)/np.max(image)
    img_gaussian = np.empty_like(image)
    for i in range(len(image)):
        img_gaussian[i,:,:] = filters.gaussian(image[i,:,:], sigma)
    return img_gaussian
