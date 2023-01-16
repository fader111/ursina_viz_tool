import cv2, numpy as np, math

def CalculateMean(points):
    result = None
    result = cv2.reduce(points, 0, cv2.REDUCE_AVG)
    return result


# cv2.Mat_<double>
def findRigidTransform(points1, points2):

    # /* Calculate centroids. */
    t1 = -CalculateMean(points1)
    t2 = -CalculateMean(points2)

    T1 = np.eye(4, 4)
    T1[0, 3] = t1[0]
    T1[1, 3] = t1[1]
    T1[2, 3] = t1[2]

    T2 = cv2.eye(4, 4)
    T2[0, 3] = -t2[0]
    T2[1, 3] = -t2[1]
    T2[2, 3] = -t2[2]

    # /* Calculate covariance matrix for input points. 
    # Also calculate RMS deviation from centroid
    #  * which is used for scale calculation.
    # cv2.Mat_<double> C(3, 3, 0.0)
    C = np.zeros((3,3))
    p1Rms = 0
    p2Rms = 0
    # for (ptIdx = 0 ptIdx < points1.rows ptIdx++) (
    for ptIdx in range(0, points1.rows):  
        p1 = points1(ptIdx, 0) + t1
        p2 = points2(ptIdx, 0) + t2
        p1Rms += p1.dot(p1)
        p2Rms += p2.dot(p2)
        # for (int i = 0 i < 3 i++) (
        for i in range(0, 3):
            # for (int j = 0 j < 3 j++) (
            for j in range(0,3):    
                C[i, j] += p2[i] * p1[j]

    w, u, vh = cv2.SVDecomp(C) #→ w, u, vh

    R = u * vh

    if cv2.determinant(R) < 0: 
        R -= u.col(2) * (vh.row(2) * 2.0)

    scale = math.sqrt(p2Rms / p1Rms)
    R *= scale

    M = np.eye(4, 4)
    R.copyTo(M[:3][:3])

    result = T2 * M * T1
    result /= result(3, 3)

    return result[:3]

