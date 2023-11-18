# Ahmet Serdar Çanak (1760039)

import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb
from skimage import data

if __name__ == "__main__":
    image = data.rocket()
    histogram_pre_processing = rgb2hsv(image)[:, :, 0]

    hsv_img = rgb2hsv(image)

    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            if hsv_img[y, x, 0] > 0.20 or ((hsv_img[y, x, 1] < 0.55) or (hsv_img[y, x, 2] < 0.8)):
                hsv_img[y, x] = [0, 0, hsv_img[y, x, 2]]

    rgb_img = hsv2rgb(hsv_img)
    histogram_post_processing = hsv_img[:, :, 0]

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))

    ax0.imshow(image)
    ax0.set_title("Original image")
    ax0.axis('off')
    ax1.imshow(rgb_img, cmap='hsv')
    ax1.set_title("Altered image")
    ax1.axis('off')
    ax2.hist(histogram_pre_processing.ravel(), 256)
    ax2.set_title("Histogram of the Original image")
    ax2.set_xbound(0, 1)
    ax3.hist(histogram_post_processing.ravel(), 256)
    ax3.set_title("Histogram of the Altered image")
    ax3.set_xbound(0, 1)
    ax3.set_ybound(0, 80)

    fig.tight_layout()
    plt.show()