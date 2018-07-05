""" Utility Functions for various geometric operations """
import numpy as np
import re


#for testing validity of axis order arguments
axis_order_re = re.compile('[XYZ][XYZ][XYZ]')


def axis_order_is_valid(order):
    """ return true if the axis order has valid form, e.g. 'XYZ'
    """
    if not axis_order_re.match(order):
        return False
    # check for repeats
    if order[0] == order[1] or order[0] == order[2]:
        return False
    if order[1] == order[2]:
        return False
    return True


def axis_angle_to_matrix(axis,theta):
    """ Convert a rotation axis / angle pair to a 3x3 rotation matrix """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis,theta))


def axis_angle_to_quaternion(axis, theta):
    """ Convert a rotation axis / angle pair to a quaternion """
    # make sure axis has unit length
    axis_u = axis / np.linalg.norm(axis)
    sin_half = np.sin(theta/2.0)
    cos_half = np.cos(theta/2.0)
    q = np.array((sin_half, sin_half, sin_half, cos_half)) * np.append(axis_u,1.0)
    return q


def axis_from_string(axis_string):
    if axis_string == 'X':
        return np.array((1,0,0))
    elif axis_string == 'Y':
        return np.array((0,1,0))
    elif axis_string == 'Z':
        return np.array((0,0,1))
    else:
        raise Exception('Expecting one of [X,Y,Z], got ' + axis_string)


def Euler_angles_to_quaternion(theta1, theta2, theta3, order='XYZ'):
    """ default order applies rotation around x axis first, y second, and z third.
    """
    if not axis_order_is_valid(order):
        raise Exception('Invalid order string: ' + str(order))

    e1 = axis_from_string(order[0])
    e2 = axis_from_string(order[1])
    e3 = axis_from_string(order[2])

    q1 = np.append(e1 * np.sin(theta1/2.0), np.cos(theta1/2.0))
    q2 = np.append(e2 * np.sin(theta2/2.0), np.cos(theta2/2.0))
    q3 = np.append(e3 * np.sin(theta3/2.0), np.cos(theta3/2.0))

    return compose_quaternions((q1,q2,q3))


def quaternion_to_Euler_angles(q, order='XYZ'):
    """ convert q to Euler angles
        angles are returned in the order of application, specified by order
        adapted and generalized based on code available at the following url:
        http://bediyap.com/programming/convert-quaternion-to-euler-rotations/
    """
    if not axis_order_is_valid(order):
        raise Exception('Invalid order string: ' + str(order))
    p0 = q[3]  # real component
    p1 = q[0]
    if order[0] == 'Y':
        p1 = q[1]
    elif order[0] == 'Z':
        p1 = q[2]
    p2 = q[1]
    if order[1] == 'X':
        p2 = q[0]
    elif order[1] == 'Z':
        p2 = q[2]
    p3 = q[2]
    if order[2] == 'X':
        p3 = q[0]
    elif order[2] == 'Y':
        p3 = q[1]

    e1 = axis_from_string(order[0])
    e2 = axis_from_string(order[1])
    e3 = axis_from_string(order[2])

    e = np.sign(np.dot(np.cross(e3,e2),e1))

    theta1 = np.arctan2(e*2*(p2*p3 + e*p0*p1), p0*p0 - p1*p1 - p2*p2 + p3*p3)
    theta2 = np.arcsin(-e*2*(p1*p3 - e*p0*p2))
    theta3 = np.arctan2(e*2*(p1*p2 + e*p0*p3), p0*p0 + p1*p1 - p2*p2 - p3*p3)

    return theta1, theta2, theta3


def quaternion_to_matrix(q):
    """ Convert a quaternion to an orthogonal rotation matrix """
    # normalize quaternion
    norm = np.sqrt((q*q).sum())
    q /= norm
    # fill in rotation matrix
    R = np.zeros((3, 3))

    # save as a,b,c,d for easier reading of conversion math
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]

    R[0,0] = 1 - 2*y*y - 2*z*z
    R[0,1] = 2*x*y - 2*z*w
    R[0,2] = 2*x*z + 2*y*w

    R[1,0] = 2*x*y + 2*z*w
    R[1,1] = 1 - 2*x*x - 2*z*z
    R[1,2] = 2*y*z - 2*x*w

    R[2,0] = 2*x*z - 2*y*w
    R[2,1] = 2*y*z + 2*x*w
    R[2,2] = 1 - 2*x*x - 2*y*y

    return R


def matrix_to_quaternion(rot):
    """ convert rotation matrix rot to quaternion
        Adapted from vnl_quaternion.txx in vxl
    """
    d0 = rot[0,0]
    d1 = rot[1,1]
    d2 = rot[2,2]
    xx = 1.0 + d0 - d1 - d2
    yy = 1.0 - d0 + d1 - d2
    zz = 1.0 - d0 - d1 + d2
    rr = 1.0 + d0 + d1 + d2

    vals = (xx,yy,zz,rr)
    imax = np.argmax(np.abs(vals))

    q_re = 0
    q_im = [0,0,0]

    if 3 == imax:
        r4 = np.sqrt(rr)*2
        q_re = r4 / 4
        ir4 = 1.0 / r4
        q_im[0] = (rot[2,1] - rot[1,2]) * ir4
        q_im[1] = (rot[0,2] - rot[2,0]) * ir4
        q_im[2] = (rot[1,0] - rot[0,1]) * ir4

    elif 0 == imax:
        x4 = np.sqrt(xx)*2
        q_im[0] = x4 / 4
        ix4 = 1.0 / x4
        q_im[1] = (rot[1,0] + rot[0,1]) * ix4
        q_im[2] = (rot[2,0] + rot[0,2]) * ix4
        q_re = (rot[2,1] - rot[1,2]) * ix4

    elif 1 == imax:
        y4 = np.sqrt(yy)*2
        q_im[1] = y4 / 4
        iy4 = 1.0 / y4
        q_im[0] = (rot[1,0] + rot[0,1]) * iy4
        q_im[2] = (rot[2,1] + rot[1,2]) * iy4
        q_re = (rot[0,2] - rot[2,0]) * iy4

    elif 2 == imax:
        z4 = np.sqrt(zz)*2
        q_im[2] = z4 / 4
        iz4 = 1.0 / z4
        q_im[0] = (rot[2,0] + rot[0,2]) * iz4
        q_im[1] = (rot[2,1] + rot[1,2]) * iz4
        q_re = (rot[1,0] - rot[0,1]) * iz4

    return np.array((q_im[0], q_im[1], q_im[2], q_re))


def compose_quaternions(quaternion_list):
    """ return the composition of a list of quaternions
    """
    qtotal = np.array((0,0,0,1))
    for q in quaternion_list:
        q1 = qtotal
        q2 = q
        re = q1[3]*q2[3] - np.dot(q1[0:3],q2[0:3])
        imag = np.cross(q1[0:3],q2[0:3]) + q2[0:3]*q1[3] + q1[0:3]*q2[3]
        qtotal = np.array((imag[0], imag[1], imag[2], re))
    return qtotal


def Euler_angles_to_matrix(theta1, theta2, theta3, order='XYZ'):
    """ Convert Euler angles to a rotation matrix. Angles are specified in the order of application.
    """
    return quaternion_to_matrix(Euler_angles_to_quaternion(theta1, theta2, theta3, order=order))


def matrix_to_Euler_angles(M, order='XYZ'):
    """ Convert a rotation matrix to Euler angles. Angles are returned in the order of application.
    """
    return quaternion_to_Euler_angles(matrix_to_quaternion(M),order=order)
