import os
import numpy as np
import matplotlib.pyplot as plt
from src.feature_extractor import load_image_gray, harris_response, detect_corners, subpixel_refine
from test_all import find_4_corners_pure_numpy

IMG_DIR = "data/raw_checkerboards"
paths = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)]
p = paths[0]

img = load_image_gray(p)
resp = harris_response(img)
raw = detect_corners(resp)
ref = subpixel_refine(img, raw)
pts = np.array(ref)[:54]

img_4 = find_4_corners_pure_numpy(pts)

plt.imshow(img, cmap='gray')
plt.plot(pts[:,0], pts[:,1], 'r.')
if img_4 is not None:
    plt.plot(img_4[:,0], img_4[:,1], 'go', markersize=10)
    for i in range(4):
        plt.text(img_4[i,0], img_4[i,1], str(i), color='yellow', fontsize=15)
plt.savefig("test_img_4.png")
print(f"Saved to test_img_4.png with corners:\n{img_4}")
