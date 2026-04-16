
#takes in array of cam vectors and cat vectors to given optimal quaternion
def davenportq(bi, ri, a):
    B = a * np.matmul(bi, ri.T)
    S = B + B.T
    z = np.array([B[1][2] - B[2][1], B[2][0] - B[0][2], B[0][1] - B[1][0]])
    S - np.identity(3) * np.trace(B)
    z = z.reshape(3, 1)
    K = np.block([
        [np.array([np.trace(B)]), z.T],
        [z, S - np.trace(B) * np.identity(3)]
    ])
    eigens = np.linalg.eig(K)
    q = eigens[1][np.argmax(eigens[0])]
    q = q / np.linalg.norm(q)
    return q;

#quatenrion to cartesian coordinates
def q_to_cart(q):
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    return R