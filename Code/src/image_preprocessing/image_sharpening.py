import cv2
import numpy as np


def sharpen_image(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened_img = float(amount + 1) * image - float(amount) * blurred
    sharpened_img = np.maximum(sharpened_img, np.zeros(sharpened_img.shape))
    sharpened_img = np.minimum(sharpened_img, 255 * np.ones(sharpened_img.shape))
    sharpened_img = sharpened_img.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened_img, image, where=low_contrast_mask)
    return sharpened_img