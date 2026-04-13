import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from src.feature_extractor import *

IMG_PATH = "data/raw_checkerboards/z7721978882611_a6684c55c9a48c1cc3bb0866a53c98a0.jpg"
GRID_SHAPE = (9, 6)

img = load_image_gray(IMG_PATH)
resp = harris_response(img)
raw = detect_corners(resp)
refined = subpixel_refine(img, raw)
ordered = order_checkerboard_corners(refined, GRID_SHAPE)

if ordered:
    pts = np.array(ordered)
    plt.imshow(img, cmap="gray")
    # Draw arrows connecting corners
    for i in range(len(pts) - 1):
        plt.arrow(pts[i,0], pts[i,1], pts[i+1,0]-pts[i,0], pts[i+1,1]-pts[i,1], 
                  color="r", alpha=0.5, head_width=5)
    plt.plot(pts[0,0], pts[0,1], "go", markersize=10, label="Start")
    plt.legend()
    plt.savefig("corner_order_test.png")
    print("Test saved to corner_order_test.png")
else:
    print("Ordering failed!")
