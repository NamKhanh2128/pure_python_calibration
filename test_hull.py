import numpy as np
import itertools
from scipy.spatial import ConvexHull

def find_4_corners(pts):
    # Convex hull
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]
    
    # All combinations of 4 points from hull
    max_area = 0
    best_quad = None
    
    for quad in itertools.combinations(hull_pts, 4):
        # Calculate area of polygon using shoelace
        # First, must order the 4 points cyclically to form a convex polygon
        # Since they are from the convex hull and we picked 4, if we sort them by angle 
        # around their centroid, they form a valid convex quad.
        quad = np.array(quad)
        c = quad.mean(axis=0)
        angles = np.arctan2(quad[:,1] - c[1], quad[:,0] - c[0])
        quad = quad[np.argsort(angles)]
        
        # Shoelace formula
        x = quad[:,0]
        y = quad[:,1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        if area > max_area:
            max_area = area
            best_quad = quad
            
    return best_quad

# test on a random synthetic trapezoid (barrel distorted)
y, x = np.mgrid[0:6, 0:9]
pts = np.column_stack([x.ravel(), y.ravel()]).astype(float)
# add some perspective
pts[:,0] = pts[:,0] * (1 + pts[:,1]*0.1)
# add some barrel
c = pts.mean(axis=0)
r2 = np.sum((pts - c)**2, axis=1)
pts = pts + (pts - c) * 0.005 * r2[:,None]

q = find_4_corners(pts)
print("Corners found:")
print(q)
