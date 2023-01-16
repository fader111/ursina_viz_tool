import numpy as np, cv2 
from functools import reduce

def get_rigid(src, dst): # Assumes both or Nx3 matrices
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    # Compute covariance   
    H = reduce(lambda s, a: s + np.outer(a[0], a[1]), zip(src - src_mean, dst - dst_mean), np.zeros((3,3)))
    u, s, v = np.linalg.svd(H)
    R = v.T.dot(u.T) # Rotation
    T = - R.dot(src_mean) + dst_mean # Translation
    return np.hstack((R, T[:, np.newaxis])) 

if __name__ == "__main__":
    point = np.array([      [0,0,0],
                            [1,0,0],
                            [1,1,0]], dtype='float32')
                            
    pointR= np.array([      [1,0,0],
                            [1,1,0],
                            [0,1,0]], dtype='float32')

    # get_rigid(point,pointR)

    shape = (1, 4, 3)
    source = np.zeros(shape, np.float32)
    
    # [x, y, z]
    source[0][0] = [857, 120, 854]
    source[0][1] = [254, 120, 855]
    source[0][2] = [256, 120, 255]
    source[0][3] = [858, 120, 255]
    target = source * 10
    
    retval, M, inliers = cv2.estimateAffine3D(source[0], source[0])
    print(f"estimateAffine3D\n { M, }")
    gr = get_rigid(source[0], source[0])
    print(f"gr {gr}")