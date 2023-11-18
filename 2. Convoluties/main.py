# Ahmet Serdar Çanak (1760039)

import matplotlib.pyplot as plt

from scipy import ndimage
from skimage import data, filters, feature

def opgave_1(image):
    mask1 = [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
    mask2 = [[-3,-3, 5],
             [-3, 0, 5],
             [-3,-3, 5]]
    mask3 = [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0,-1]]

    image_filter1 = ndimage.convolve(image, mask1)
    image_filter2 = ndimage.convolve(image, mask2)
    image_filter3 = ndimage.convolve(image, mask3)

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                             figsize=(8, 8))
    axes = axes.ravel()

    axes[0].imshow(image, cmap=plt.cm.gray)
    axes[0].set_title('Original image')

    axes[1].imshow(image_filter1, cmap=plt.cm.gray)
    axes[1].set_title('Sobel Edge Detection')

    axes[2].imshow(image_filter2, cmap=plt.cm.gray)
    axes[2].set_title('Kirsch Edge Detection')

    axes[3].imshow(image_filter3, cmap=plt.cm.gray)
    axes[3].set_title('Robinson Edge Detection')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def opgave_2(image):
    edge_sobel  = filters.sobel_h(image)
    edge_scharr = filters.scharr_h(image)
    edge_farid  = filters.farid_h(image)

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                             figsize=(8, 8))
    axes = axes.ravel()

    axes[0].imshow(image, cmap=plt.cm.gray)
    axes[0].set_title('Original image')

    axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
    axes[1].set_title('Sobel Edge Detection')

    axes[2].imshow(edge_scharr, cmap=plt.cm.gray)
    axes[2].set_title('Scharr Edge Detection')

    axes[3].imshow(edge_farid, cmap=plt.cm.gray)
    axes[3].set_title('Farid Edge Detection')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Wat valt je op als je de resultaten van de eerste opdracht met de tweede opdracht vergelijkt?
    # Wat mij opvalt is dat er eigenlijk veel meer gedaan wordt bij de standaardfuncties dan het toepassen van een edge detection filter.

def opgave_3(image):
    # Compute the Canny filter for two values of sigma
    edges1 = feature.canny(image)
    edges2 = feature.canny(image, sigma=3)

    # display results
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)

    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('noisy image', fontsize=20)

    ax2.imshow(edges1, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title(r'Canny filter, $\sigma=1$', fontsize=20)

    ax3.imshow(edges2, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title(r'Canny filter, $\sigma=3$', fontsize=20)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image = data.grass()

    opgave_1(image)
    opgave_2(image)
    opgave_3(image)