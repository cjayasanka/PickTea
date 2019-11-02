# steps of sift process
# 1.Scale-Space Extrema Detection
# 2.Keypoint Localization
# 3.Orientation Assignment
# 4.Local Descriptor Creation

from PIL import Image
from scipy import ndimage
import math


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


if __name__ == "__main__":
    print("Hello")
    # open image
    img = Image.open('data/field.jpg').convert('LA')
    img_size = img.size
    print(img_size)

    # image size is doubled to get more key points as described in paper
    img_2x = img.resize((img_size[0] * 2, img_size[1] * 2), 2)
