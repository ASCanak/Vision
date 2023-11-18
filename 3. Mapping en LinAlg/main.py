# Ahmet Serdar Çanak (1760039)

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, transform

image = data.rocket()

rotation = transform.AffineTransform(rotation=np.pi/10,)
rotated_image = transform.warp(image, rotation)

translation = transform.AffineTransform(translation=(-75, 50))
translated_image = transform.warp(image, translation)

stretch = transform.AffineTransform(scale=(3.5, 1))
stretched_image = transform.warp(image, stretch)

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                         figsize=(8, 8))
axes = axes.ravel()

axes[0].imshow(image)
axes[0].set_title('Original image')

axes[1].imshow(rotated_image)
axes[1].set_title('After Rotation')

axes[2].imshow(translated_image)
axes[2].set_title('After Translation')

axes[3].imshow(stretched_image)
axes[3].set_title('After Stretch')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()