from PIL import Image
import numpy as np

img = Image.open("909.jpg")

img = np.asarray(img)
img = img[:,:,::-1]               #rgb to bgr
print(img[0][0][:])
print(img.shape)

print(img[0][0][:])
img.tofile("909_bgr.bin")
