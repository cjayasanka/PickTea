# steps of sift process
# 1.Scale-Space Extrema Detection
# 2.Keypoint Localization
# 3.Orientation Assignment
# 4.Local Descriptor Creation

from PIL import Image
from scipy import ndimage
import math
import numpy as np


def g_filter(sigma):
    window = 2 * np.ceil(3 * sigma) + 1
    x, y = np.mgrid[-window // 2 + 1:window // 2 + 1, -window // 2 + 1:window // 2 + 1]
    top = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) / (2 * np.pi * sigma**2)
    return top / top.sum()


# 1st step of sift
def compute_scale_space(img):

    sigma = 0.5
    scale_space = []
    img_size = img.size
    # img = img

    for i in range(4):
        octave = []
        for j in range(5):
            octave.append(ndimage.gaussian_filter(img, sigma*(math.sqrt(2))**(j+1)))
        sigma = sigma * 2
        scale_space.append(octave)
        img = img.resize((img_size[0]//2, img_size[1]//2), 2)

    return scale_space


def compute_dog(scale_space):
    # subtract from each array and return 4 resulting arrays
    dog = []
    for i in range(4):
        dog_in_octave = []
        for j in range(4):
            dog_in_octave.append(scale_space[i][j] - scale_space[i][j+1])
        dog.append(dog_in_octave)
    return dog


# extrema detection and keypoints detection
def find_keypoints(dog):
    keypoints = []
    # extrema = []
    for scale in dog:
        scale_extrema = []
        for s in range(2):
            img_size = scale[0].size
            # blank = Image.fromarray(np.zeros(scale[0].size, dtype=np.uint8))
            for i in range(img_size[0]):
                for j in range(img_size[1]):
                    if is_min_max(scale, s+1, i, j, img_size):
                        # blank[i, j] = scale[s+1][i, j]
                        scale_extrema.append([i, j, s+1])

            # scale_extrema.append(blank)

        # extrema.append(scale_extrema)
        scale_keypoints = subpixel(scale_extrema, scale)
        keypoints.append(scale_keypoints)

    return keypoints


def is_min_max(scale, img_no, i, j, img_size):
    value = scale[img_no][i, j]
    high = False
    low = False
    if value > scale[img_no-1][i, j]:
        # low scale is lower
        high = True
    elif value < scale[img_no-1][i, j]:
        # low scale is higher
        low = True
    else:
        return False

    if high:
        if value < scale[img_no+1][i, j]:
            # high scale is higher - not max or min
            return False
        # same scale level
        if value < scale[img_no][i+1, j] or value < scale[img_no][i-1, j]:
            return False
        if value < scale[img_no][i, j+1] or value < scale[img_no][i, j-1]:
            return False
        if value < scale[img_no][i+1, j+1] or value < scale[img_no][i+1, j-1]:
            return False
        if value < scale[img_no][i-1, j+1] or value < scale[img_no][i-1, j-1]:
            return False
        # one scale lower
        if value < scale[img_no-1][i + 1, j] or value < scale[img_no-1][i - 1, j]:
            return False
        if value < scale[img_no-1][i, j + 1] or value < scale[img_no-1][i, j - 1]:
            return False
        if value < scale[img_no-1][i + 1, j + 1] or value < scale[img_no-1][i + 1, j - 1]:
            return False
        if value < scale[img_no-1][i - 1, j + 1] or value < scale[img_no-1][i - 1, j - 1]:
            return False
        # one scale higher
        if value < scale[img_no + 1][i + 1, j] or value < scale[img_no + 1][i - 1, j]:
            return False
        if value < scale[img_no + 1][i, j + 1] or value < scale[img_no + 1][i, j - 1]:
            return False
        if value < scale[img_no + 1][i + 1, j + 1] or value < scale[img_no + 1][i + 1, j - 1]:
            return False
        if value < scale[img_no + 1][i - 1, j + 1] or value < scale[img_no + 1][i - 1, j - 1]:
            return False

    if low:
        if value > scale[img_no+1][i, j]:
            # high scale is lower - not max or min
            return False
        # same scale level
        if value > scale[img_no][i + 1, j] or value > scale[img_no][i - 1, j]:
            return False
        if value > scale[img_no][i, j + 1] or value > scale[img_no][i, j - 1]:
            return False
        if value > scale[img_no][i + 1, j + 1] or value > scale[img_no][i + 1, j - 1]:
            return False
        if value > scale[img_no][i - 1, j + 1] or value > scale[img_no][i - 1, j - 1]:
            return False
        # one scale lower
        if value > scale[img_no - 1][i + 1, j] or value > scale[img_no - 1][i - 1, j]:
            return False
        if value > scale[img_no - 1][i, j + 1] or value > scale[img_no - 1][i, j - 1]:
            return False
        if value > scale[img_no - 1][i + 1, j + 1] or value > scale[img_no - 1][i + 1, j - 1]:
            return False
        if value > scale[img_no - 1][i - 1, j + 1] or value > scale[img_no - 1][i - 1, j - 1]:
            return False
        # one scale higher
        if value > scale[img_no + 1][i + 1, j] or value > scale[img_no + 1][i - 1, j]:
            return False
        if value > scale[img_no + 1][i, j + 1] or value > scale[img_no + 1][i, j - 1]:
            return False
        if value > scale[img_no + 1][i + 1, j + 1] or value > scale[img_no + 1][i + 1, j - 1]:
            return False
        if value > scale[img_no + 1][i - 1, j + 1] or value > scale[img_no + 1][i - 1, j - 1]:
            return False

    return True


if __name__ == "__main__":
    print("Hello")
    # open image
    img = Image.open('data/field.jpg').convert('LA')
    img_size = img.size
    print(img_size)

    # image size is doubled to get more key points as described in paper
    img_2x = img.resize((img_size[0] * 2, img_size[1] * 2), 2)
