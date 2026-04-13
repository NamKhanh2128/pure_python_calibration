import numpy as np
import itertools

def find_4_corners_pure_numpy(pts):
    # 1. Collect points that are extreme in multiple orientations
    hull_candidates = set()
    angles = np.linspace(0, np.pi/2, 10)
    
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        # project all points onto this axis and perpendicular axis
        proj_x = pts[:,0]*c + pts[:,1]*s
        proj_y = -pts[:,0]*s + pts[:,1]*c
        
        hull_candidates.add(np.argmin(proj_x))
        hull_candidates.add(np.argmax(proj_x))
        hull_candidates.add(np.argmin(proj_y))
        hull_candidates.add(np.argmax(proj_y))
        
    candidates = list(hull_candidates)
    
    max_area = 0
    best_quad = None
    
    for quad_idx in itertools.combinations(candidates, 4):
        quad = pts[list(quad_idx)]
        
        # order cyclically
        centroid = quad.mean(axis=0)
        angles_c = np.arctan2(quad[:,1] - centroid[1], quad[:,0] - centroid[0])
        quad = quad[np.argsort(angles_c)]
        
        # area via shoelace
        x, y = quad[:,0], quad[:,1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        if area > max_area:
            max_area = area
            best_quad = quad
            
    return best_quad

# test it
y, x = np.mgrid[0:6, 0:9]
pts = np.column_stack([x.ravel(), y.ravel()]).astype(float)
pts[:,0] = pts[:,0] * (1 + pts[:,1]*0.1)
c = pts.mean(axis=0)
r2 = np.sum((pts - c)**2, axis=1)
pts = pts + (pts - c) * 0.005 * r2[:,None]

print(find_4_corners_pure_numpy(pts))
