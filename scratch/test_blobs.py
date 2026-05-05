import glob
import numpy as np
from PIL import Image

def find_blobs(img_path):
    # Load and threshold
    img = np.array(Image.open(img_path).convert("L"))
    
    # Simple adaptive threshold or fixed threshold
    thresh = np.mean(img) * 0.8
    binary = img < thresh  # Dark objects are True
    
    # We can use a simple connected components algorithm (pure numpy/python)
    # Since we can't use scipy/cv2, let's write a simple flood fill
    H, W = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    
    blobs = []
    
    # To speed up, we can sub-sample or just do it in python
    # But wait, flood fill in python can hit recursion limits.
    # Let's use a queue
    for r in range(H):
        for c in range(W):
            if binary[r, c] and not visited[r, c]:
                # BFS
                q = [(r, c)]
                visited[r, c] = True
                area = 0
                sum_r, sum_c = 0, 0
                
                head = 0
                while head < len(q):
                    curr_r, curr_c = q[head]
                    head += 1
                    area += 1
                    sum_r += curr_r
                    sum_c += curr_c
                    
                    # neighbors
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if binary[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                
                if area > 50: # filter noise
                    blobs.append((sum_c / area, sum_r / area, area))
                    
    return blobs

paths = glob.glob('data/raw_checkerboards/*.jpg')
if paths:
    blobs = find_blobs(paths[0])
    print(f"Found {len(blobs)} blobs")
    # Sort by area to see
    blobs.sort(key=lambda x: x[2], reverse=True)
    print("Top 5 areas:", [b[2] for b in blobs[:5]])
    # Print number of blobs in a reasonable area range
    median_area = np.median([b[2] for b in blobs[:150]])
    valid_blobs = [b for b in blobs if median_area * 0.2 < b[2] < median_area * 5]
    print(f"Valid blobs: {len(valid_blobs)}")
