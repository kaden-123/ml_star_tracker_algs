import numpy as np

#takes in array of cam vectors and cat vectors to given optimal quaternion
# def davenportq(bi, ri, a):
    
#     B = a * np.matmul(bi, ri.T)
#     S = B + B.T
#     z = np.array([B[1][2] - B[2][1], B[2][0] - B[0][2], B[0][1] - B[1][0]])
#     S - np.identity(3) * np.trace(B)
#     z = z.reshape(3, 1)
#     K = np.block([
#         [np.array([np.trace(B)]), z.T],
#         [z, S - np.trace(B) * np.identity(3)]
#     ])
#     eigens = np.linalg.eig(K)
#     q = eigens[1][np.argmax(eigens[0])]
#     q = q / np.linalg.norm(q)
#     return q;

# most formulas taken from here https://www.vectornav.com/resources/inertial-navigation-primer/math-fundamentals/math-attitudetran

def e_to_q(euler):
    """
    Quaternion [w, x, y, z] from 3-2-1 Euler [roll, pitch, yaw]
    (rotation order: ZYX)
    """
    phi, theta, psi = euler  # roll, pitch, yaw

    c1 = np.cos(phi / 2)
    s1 = np.sin(phi / 2)

    c2 = np.cos(theta / 2)
    s2 = np.sin(theta / 2)

    c3 = np.cos(psi / 2)
    s3 = np.sin(psi / 2)

    w = c1*c2*c3 + s1*s2*s3
    x = s1*c2*c3 - c1*s2*s3
    y = c1*s2*c3 + s1*c2*s3
    z = c1*c2*s3 - s1*s2*c3

    return np.array([w, x, y, z])

def q_to_e(q):
    """
    Returns Euler angles [roll, pitch, yaw] (3-2-1 / ZYX)
    """
    w, x, y, z = q

    # roll (x)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # yaw (z)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

def q_to_DCM(q):
    """ given a quaternion in the form np.array([w, x, y, z]) return DCM as numpy mat """
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    return R

def DCM_to_q(DCM):
    """ given a DCM as numpy mat, return a quateron in the for np.array([w, x, y, z]) """
    t_one = DCM[0][0]
    t_two = DCM[1][1]
    t_three = DCM[2][2]
    
    x = np.sqrt((1 + t_one - t_two - t_three)) / 2
    y = np.sqrt((1 - t_one + t_two - t_three)) / 2
    z = np.sqrt((1 - t_one - t_two + t_three)) / 2
    w = np.sqrt((1 + t_one + t_two + t_three)) / 2

    return np.array([w, x, y, z])

def e_to_DCM(euler):
    """DCM from 3-2-1 Euler angles [roll, pitch, yaw]"""

    phi, theta, psi = euler

    s_phi = np.sin(phi)
    c_phi = np.cos(phi)
    s_theta = np.sin(theta)
    c_theta = np.cos(theta)
    s_psi = np.sin(psi)
    c_psi = np.cos(psi)

    return np.array([
        [
            c_theta * c_psi,
            c_theta * s_psi,
            -s_theta
        ],
        [
            s_phi * s_theta * c_psi - c_phi * s_psi,
            s_phi * s_theta * s_psi + c_phi * c_psi,
            s_phi * c_theta
        ],
        [
            c_phi * s_theta * c_psi + s_phi * s_psi,
            c_phi * s_theta * s_psi - s_phi * c_psi,
            c_phi * c_theta
        ]
    ])

def DCM_to_e(DCM):
    """ returns a np array roll pitch yaw (3 - 2 - 1) from given DCM mat """
    theta = np.arcsin(DCM[0, 2])
    phi = np.arctan2(-DCM[1, 2], DCM[2, 2])  
    psi = np.arctan2(-DCM[0, 1], DCM[0, 0])   

    return np.array([phi, theta, psi])

def DCM(e, d):
    """ given two lists 3d basis np vectors, return DCM for D -> E """
    D = np.column_stack([d[0], d[1], d[2]])
    E = np.column_stack([e[0], e[1], e[2]])  
    DCM = np.matmul(D.T, E)

    return DCM
    