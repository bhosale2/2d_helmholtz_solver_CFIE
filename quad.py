import numpy as np
import numpy.linalg as la
import scipy.special
import matplotlib.pyplot as plt
import scipy.special as scp

# import scipy.sparse.linalg as spla

cos = np.cos
sin = np.sin
pi = np.pi

delta = 0.8
# delta = 1e-3


def curve(t):
    # return np.array([(3/4)*cos(t-pi/4.)*(1 + sin(2*t)/2),
    #                 sin(t-pi/4)*(1 + sin(2*t)/2)])
    # paper shape
    return np.array(
        [
            1.1 + (1 + 0.2 * cos(4 * t)) * cos(t),
            1.2 + delta + (1 + 0.2 * cos(4 * t)) * sin(t),
        ]
    )
    # return np.array([(1 + 0.2 * cos(4*t)) * cos(t),
    #                  (1 + 0.2 * cos(4*t)) * sin(t)])


def dcurve_dt(t):
    # return np.array([
    #     -(3./4.)*sin(t-pi/4.)*(1 + sin(2*t)/2.) +
    #     (3./4.)*cos(t-pi/4.)*cos(2*t),
    #     cos(t-pi/4.)*(1 + sin(2*t)/2.) + sin(t-pi/4.)*cos(2*t)
    #     ])
    return np.array(
        [
            -(1 + 0.2 * cos(4 * t)) * sin(t) - 0.8 * sin(4 * t) * cos(t),
            (1 + 0.2 * cos(4 * t)) * cos(t) - 0.8 * sin(4 * t) * sin(t),
        ]
    )


def u_exact(points, k):
    """
    evaluates exact solution by point source
    """
    x, y = points
    # for validation
    return scp.hankel1(0, k * np.sqrt((x - 1.1) ** 2 + (y - 1.1 - delta) ** 2)) * 0.25j
    # for scattering source
    # return (scp.hankel1(0, k*np.sqrt((x+2)**2 + (y-2)**2)) * 0.25j
    #         - scp.hankel1(0, k*np.sqrt((x+2)**2 + (y+2)**2)) * 0.25j)

    # for plane wave scattering
    # return np.exp(x * k * 1j)


test_targets = np.array([[-0.2, 0], [0.2, 0], [0, -0.2], [0, 0.2]]).T


npanels1 = 10
npanels2 = 20


# This data structure helps you get started by setting up geometry
# and Gauss quadrature panels for you.


class QuadratureInfo:
    def __init__(self, nintervals, **kwargs):
        self.nintervals = nintervals
        # par_length = 2*np.pi
        intervals = np.linspace(0, 2 * np.pi, nintervals + 1)
        self.npoints = 7 + 1
        self.shape = (nintervals, self.npoints)

        ref_info = scipy.special.legendre(self.npoints).weights
        ref_nodes = ref_info[:, 0]
        ref_weights = ref_info[:, 2]

        par_intv_length = intervals[1] - intervals[0]

        self.par_nodes = np.zeros((nintervals, self.npoints))
        for i in range(nintervals):
            a, b = intervals[i : i + 2]

            assert abs((b - a) - par_intv_length) < 1e-10
            self.par_nodes[i] = ref_nodes * par_intv_length * 0.5 + (b + a) * 0.5

        self.curve_nodes = curve(self.par_nodes.reshape(-1)).reshape(2, nintervals, -1)
        self.curve_deriv = dcurve_dt(self.par_nodes.reshape(-1)).reshape(
            2, nintervals, -1
        )

        self.curve_speed = la.norm(self.curve_deriv, 2, axis=0)

        tangent = self.curve_deriv / self.curve_speed
        tx, ty = tangent
        self.normals = np.array([ty, -tx])

        self.curve_weights = self.curve_speed * ref_weights * par_intv_length / 2
        self.panel_lengths = np.sum(self.curve_weights, 1)

        # if 0:
        if kwargs["show_bound"] == 1:
            plt.plot(self.curve_nodes[0].reshape(-1), self.curve_nodes[1].reshape(-1), "g")
        # plt.quiver(self.curve_nodes[0], self.curve_nodes[1],
        #            self.normals[0], self.normals[1])
        # plt.show()
