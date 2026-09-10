import numpy as np
# most formulas taken from here https://www.vectornav.com/resources/inertial-navigation-primer/math-fundamentals/math-attitudetran
# also note that all inputted angles are in radians


def e_to_q(euler):
    """ returns numpy [w, x, y, z] given euler 3 2 1 """
    phi, theta, psi = euler  # roll, pitch, yaw

    c_phi = np.cos(phi / 2)
    s_phi = np.sin(phi / 2)

    c_theta = np.cos(theta / 2)
    s_theta = np.sin(theta / 2)

    c_psi = np.cos(psi / 2)
    s_psi = np.sin(psi / 2)

    w = c_phi * c_theta * c_psi + s_phi * s_theta * s_psi
    x = s_phi * c_theta * c_psi - c_phi * s_theta * s_psi
    y = c_phi * s_theta * c_psi + s_phi * c_theta * s_psi
    z = c_phi * c_theta * s_psi - s_phi * s_theta * c_psi

    return np.array([w, x, y, z])

def q_to_e(q):
    """ returns np array euler angles 3 2 1 given normalized quaternion w, x, y, z"""
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x*x + y*y))
    pitch = (-np.pi / 2) + (2 * np.arctan2(np.sqrt(1 + 2 * (w * y - x * z)), np.sqrt(1 - 2 * (w * y - x * z))))
    yaw = np.arctan2( 2 * (w * z + x * y), 1 - 2 * (y*y + z*z))

    return np.array([roll, pitch, yaw])

def q_to_DCM(q):
    """ given a quaternion w, x, y, z return DCM as numpy mat """
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    return R.T

def DCM_to_q(DCM):
    """ given a DCM as numpy mat, return a quaternion np array w, x, y, z """
    w2 = (1 + DCM[0][0] + DCM[1][1] + DCM[2][2]) / 4
    x2 = (1 + DCM[0][0] - DCM[1][1] - DCM[2][2]) / 4
    y2 = (1 - DCM[0][0] + DCM[1][1] - DCM[2][2]) / 4
    z2 = (1 - DCM[0][0] - DCM[1][1] + DCM[2][2]) / 4

    largest = np.argmax([w2, x2, y2, z2])

    match largest:
        case 0:  
            w = np.sqrt(w2)
            x = (DCM[1][2] - DCM[2][1]) / (4*w)  
            y = (DCM[2][0] - DCM[0][2]) / (4*w) 
            z = (DCM[0][1] - DCM[1][0]) / (4*w)  
        case 1:  
            x = np.sqrt(x2)
            w = (DCM[1][2] - DCM[2][1]) / (4*x)  
            y = (DCM[0][1] + DCM[1][0]) / (4*x)  
            z = (DCM[2][0] + DCM[0][2]) / (4*x)  
        case 2: 
            y = np.sqrt(y2)
            w = (DCM[2][0] - DCM[0][2]) / (4*y)  
            x = (DCM[0][1] + DCM[1][0]) / (4*y)  
            z = (DCM[1][2] + DCM[2][1]) / (4*y)  
        case 3:  
            z = np.sqrt(z2)
            w = (DCM[0][1] - DCM[1][0]) / (4*z)  
            x = (DCM[2][0] + DCM[0][2]) / (4*z)  
            y = (DCM[1][2] + DCM[2][1]) / (4*z)

    return np.array([w, x, y, z]) / np.linalg.norm([w, x, y, z])

def e_to_DCM(euler):
    """np mat DCM from 3-2-1 Euler angles"""

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
    """ returns a np array 3 2 1 from given DCM mat (DCM must be ortho) """
    theta = np.arcsin(-DCM[0, 2])
    phi = np.arctan2(DCM[1, 2], DCM[2, 2])  
    psi = np.arctan2(DCM[0, 1], DCM[0, 0])   

    return np.array([phi, theta, psi])

def DCM(e, d):
    """ given two lists 3d orthonormal basis np vectors, return DCM for D -> E """
    D = np.column_stack([d[0], d[1], d[2]])
    E = np.column_stack([e[0], e[1], e[2]])  
    DCM = np.matmul(E.T, D)

    return DCM

def q_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
    
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
    
    
    