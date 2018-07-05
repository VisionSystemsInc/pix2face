""" camera decomposition methods """
import numpy as np
import scipy.linalg
from . import geometry_utils


def decompose_affine(P, verbose=False):
    """ decompose affine projection matrix into K,D,R, and T such that:

    P = K*D*[R | T]
            [0 | 1]

    D = [1 0 0 0]
        [0 1 0 0]
        [0 0 0 1]
    """
    A = P[0:2,0:3]
    T = P[0:2,3]
    AR,AQ = scipy.linalg.rq(A,mode='economic')

    # make sure that diagonal elements of AR are positive
    S = np.diag(np.sign(np.diag(AR)))
    AR = np.dot(AR,S)
    AQ = np.dot(S,AQ)

    K = np.eye(3)
    K[0:2,0:2] = AR

    R2 = AQ
    T2 = np.dot(np.linalg.inv(AR), P[0:2,3])

    rotx = R2[0,:]
    roty = R2[1,:]
    rotz = np.cross(rotx, roty)

    R = np.vstack((rotx, roty, rotz))
    T = np.array((T2[0], T2[1], 0))

    # 4x4 homogeneous extrinsic transformation
    RT = np.vstack(( np.hstack( (R,T.reshape(3,1)) ), (0,0,0,1) ))

    DropZ = np.zeros((3,4))
    DropZ[0:2,0:2] = np.eye(2)
    DropZ[2,3] = 1

    P2 = np.dot(  np.dot(K, DropZ), RT  )
    residual = np.abs(P - P2)
    if verbose:
        print(str(residual))
    if np.any(residual > 1e-6):
        raise Exception('Error recomposing projection matrix: residual = ' + str(residual))

    return K,DropZ,R,T


def affine_to_orthographic(K,R,T, limit_H_diagonal=True):
    """ factor out shear and stretch from K, return new K,R,T,H s.t.
    H may be applied to the 3-d points to account for shear and warp
    """
    # construct orthographic projection matrix, and transform A mapping K to Kortho
    Kortho = np.diag((K[0,0], K[0,0], 1))
    A3x3 = np.dot(np.linalg.inv(K), Kortho)
    A = np.eye(4)
    A[0:3,0:3] = A3x3

    # 4x4 homogeneous extrinsic transformation
    RT = np.vstack(( np.hstack( (R,T.reshape(3,1)) ), (0,0,0,1) ))

    # now construct transform H that can be applied to 3-d points to account for A
    Ainv = np.linalg.inv(A)
    RTinv = np.linalg.inv(RT)
    H = np.dot(np.dot(RTinv, Ainv), RT)

    DropZ = np.zeros((3,4))
    DropZ[0:2,0:2] = np.eye(2)
    DropZ[2,3] = 1

    # verify that decomposition gives back the original projection matrix P
    P = np.dot( np.dot( K,DropZ), RT)
    P3 = np.dot( np.dot( np.dot(Kortho, DropZ), RT), H)
    if np.any(np.abs(P - P3) > 1e-6):
        raise Exception('Error recomposing projection matrix')

    # factor rotation component out of H and apply to extrinsics
    H3x3Q,H3x3R = np.linalg.qr(H[0:3,0:3])
    S = np.diag(np.sign(np.diag(H3x3R)))
    H3x3Q = np.dot(H3x3Q,S)
    H3x3R = np.dot(S,H3x3R)

    if limit_H_diagonal:
        # zero out off-diagonal elements
        H3x3R = np.diag(np.diag(H3x3R))

    HQ = np.eye(4)
    HQ[0:3,0:3] = H3x3Q
    HQ[0:3,3] = H[0:3,3]
    Hortho = np.eye(4)
    Hortho[0:3,0:3] = H3x3R

    RTortho = np.dot(RT,HQ)

    return Kortho, RTortho[0:3,0:3], RTortho[0:3,3], Hortho


def decompose_camera_rotation(camR, pitch_offset=0):
    """ decompose rotation matrix into yaw, pitch, roll (units of degrees)
    """
    R = np.dot(np.diag((1,-1,-1)), camR)
    euler_angles = geometry_utils.matrix_to_Euler_angles(R, order='YXZ')
    yaw = np.rad2deg(euler_angles[0])
    pitch = np.rad2deg(euler_angles[1]) + pitch_offset
    roll = np.rad2deg(euler_angles[2])

    return yaw, pitch, roll


def compose_camera_rotation(yaw_deg, pitch_deg, roll_deg, pitch_offset=0):
    """ compose camera rotation matrix from yaw, pitch, roll (units of degrees)
    """
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg - pitch_offset)
    roll = np.deg2rad(roll_deg)
    R = geometry_utils.Euler_angles_to_matrix(yaw,pitch,roll, order='YXZ')
    Rcam = np.dot(np.diag((1,-1,-1)),R)

    return Rcam


def projection_error(camera, points_2d, points_3d):
    """ return the mean projection error over all points """
    pts_projected = camera.project_points(points_3d)
    errors = np.array(pts_projected) - np.array(points_2d)
    error_mags = np.sqrt(np.sum(errors * errors, axis=1))
    mean_error = np.mean(error_mags, axis=0)
    return mean_error


