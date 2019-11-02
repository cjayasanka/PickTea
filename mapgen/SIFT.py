# steps of sift process
# 1.Scale-Space Extrema Detection
# 2.Keypoint Localization
# 3.Orientation Assignment
# 4.Local Descriptor Creation

from PIL import Image

if __name__ == "__main__":
    print("Hello")
    # open image
    img = Image.open('data/field.jpg').convert('LA')
    img_size = img.size
    print(img_size)
