import numpy as np
from scipy.spatial.distance import cdist


def transform_calc(A, B):
    '''
    Calculates the least-squares best-fit transform between corresponding 3D points A->B
    Input:
      A: Nx3 numpy array of corresponding 3D points
      B: Nx3 numpy array of corresponding 3D points
    Returns:
      T: 4x4 homogeneous transformation matrix
      R: 3x3 rotation matrix
      t: 3x1 column vector
    '''

    assert len(A) == len(B)

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t


def icp(A, B, max_iterations=20, d_max=0.001):
    """[summary]

    Args:
        A : Nx3 : ([numpy array]): [array of source 3D points]
        B: Nx3 numpy array of destination 3D point

        max_iterations (int, optional): [this is to exit the algorithm]. Defaults to 20.
        d_max (float, optional): [convergence criteria, low value == bad convergence]. Defaults to 0.001.
    Input : 
        A = {a_i} and B = {b_i}, two point clouds
    Output:
        T: final homogeneous transformation which aligns A and B
        distances: Euclidean distances (errors) of the nearest neighbor
    """
    # creating the src and dst variables
    # copying the input A and B point cloud into in src and dst variables
    # copying to maintain the original element of input A and B as it is. (not making changes in them)
    src = np.ones((4, A.shape[0]))
    dst = np.ones((4, B.shape[0]))
    src[0:3, :] = np.copy(A.T)
    dst[0:3, :] = np.copy(B.T)

    # apply the initial pose estimation
    T0 = np.identity(4)
    src = np.dot(T0, src)
    # error is set to 0
    prev_error = 0

    for i in range(max_iterations):

        ######Calculate the closest point in the triangular mesh####

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.sum(distances) / distances.size
        if abs(prev_error-mean_error) < tolerance:
            break
        prev_error = mean_error
