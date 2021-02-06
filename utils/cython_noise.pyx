from skimage.color import rgb2lab
import numpy as np
import random

def pepper(img, amount=0.5):
    _range = 0.4
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if random.random() <= amount:
                img[:, i, j] = img[:, i, j] * random.random() * _range

    return img


# def salt(img, amount=0.5):
#     _range = 0.7
#     h, w, _ = img.shape
#     rand_img = np.random.rand(h, w)
#     white_mask = rand_img <= amount
#     out = img

#     luminosity = (np.random.rand(*out.shape[:-1]) * _range) + 1 - _range

#     luminosity = np.dstack([luminosity] * 3)
#     out[white_mask, :] = out[white_mask, :] + luminosity[white_mask, :]
#     out = np.clip(out, 0, 1)
#     return out


# def gaussian(img, amount=0.2, calibration=0.05):
#     h, w, ch = img.shape
#     noise = np.random.normal(0, amount, (h, w, ch)) - calibration
#     out = img
#     out = out + noise
#     out = np.clip(out, 0, 1)
#     return out
