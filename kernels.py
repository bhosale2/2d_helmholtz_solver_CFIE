from sympy import symbols, lambdify, diff, sqrt, I
from sympy import besselj, hankel1, atan2, exp, pi, tanh
import scipy.special as scp
import numpy as np
from scipy.sparse.linalg import gmres


# gmres iteration counter
# https://stackoverflow.com/questions/33512081/getting-the-number-of-iterations-of-scipys-gmres-iterative-method
class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1


counter = gmres_counter()


def sommerfeld(k, beta, interval, exp_type):
    """
    evaluates Sommerfeld term either full or partial based on argument provided
    """
    x_1, y_1, x_2, y_2, ny_1, ny_2 = symbols('x_1, y_1, x_2, y_2, ny_1, ny_2')
    som_int_sp = 0
    t_max = k + interval
    for t in range(-t_max, t_max + 1):
        lam = t - I * tanh(t)
        d_lam = 1 - I * (1 - tanh(t) ** 2)
        f = (lam ** 2 - k ** 2) ** 0.5
        if exp_type == "full":
            term = (exp(-f * (x_2 + y_2)) * exp(I * lam * (x_1 - y_1)) *
                    (f + I * beta) / (f - I * beta) / f * d_lam / 4 / pi)
        else:
            term = (exp(-f * (x_2 + y_2)) * exp(I * lam * (x_1 - y_1)) *
                    exp(-f + I * beta) / (f - I * beta) / f * d_lam / 4 / pi)
        if (t == -t_max or t == t_max):
            som_int_sp += 0.5 * term
        else:
            som_int_sp += term
    som_sp = lambdify([x_1, x_2, y_1, y_2], som_int_sp)
    som_int_dp = ny_1 * diff(som_int_sp, y_1) + ny_2 * diff(som_int_sp, y_2)
    som_dp = lambdify([x_1, x_2, y_1, y_2, ny_1, ny_2], som_int_dp)
    return som_sp, som_dp


def eval_sp_dp_QBX(order, k):
    """
    evaluates single layer, double layer and corr QBX coeffs for Helmholtz
    calculates image potential for y = -y - eta as well
    """
    x_1, y_1, x_2, y_2, eta = symbols('x_1, y_1, x_2, y_2, eta')
    nx_1, nx_2, ny_1, ny_2, r = symbols('nx_1, nx_2, ny_1, ny_2, r')
    dist = sqrt((x_1 - y_1) ** 2 + (x_2 - y_2) ** 2)
    kernel = I / 4 * hankel1(0, k * dist)
    single_layer = lambdify([x_1, x_2, y_1, y_2], kernel)
    green_normal_der = ny_1 * diff(kernel, y_1) + ny_2 * diff(kernel, y_2)
    double_layer = lambdify([x_1, x_2, y_1, y_2,
                            ny_1, ny_2], green_normal_der)
    # image in y=0 calculations
    image_dist = sqrt((x_1 - y_1) ** 2 + (x_2 + y_2 + eta) ** 2)
    image_kernel = I / 4 * hankel1(0, k * image_dist)
    image_single_layer = lambdify([x_1, x_2, y_1, y_2, eta], image_kernel)
    image_green_normal_der = (ny_1 * diff(image_kernel, y_1) +
                              ny_2 * diff(image_kernel, y_2))
    image_double_layer = lambdify([x_1, x_2, y_1, y_2, eta,
                                  ny_1, ny_2], image_green_normal_der)
    # Grafs theorem term evaluations
    c_1 = x_1 + nx_1 * r
    c_2 = x_2 + nx_2 * r
    xc = sqrt((x_1 - c_1) ** 2 + (x_2 - c_2) ** 2)
    yc = sqrt((y_1 - c_1) ** 2 + (y_2 - c_2) ** 2)
    x_theta = atan2((x_2 - c_2), (x_1 - c_1))
    y_theta = atan2((y_2 - c_2), (y_1 - c_1))
    img_yc = sqrt((y_1 - c_1) ** 2 + (-(y_2 + eta) - c_2) ** 2)
    img_y_theta = atan2((-(y_2 + eta) - c_2), (y_1 - c_1))
    # single layer expansion zeroth order term
    qbx_exp_slp = I / 4 * hankel1(0, k * yc) * besselj(0, k * xc)
    img_qbx_exp_slp = I / 4 * hankel1(0, k * img_yc) * besselj(0, k * xc)
    for i in range(1, order+1):
        qbx_exp_slp += I / 4 * (hankel1(i, k * yc) * exp(I * i * y_theta)
                                * besselj(i, k * xc) * exp(-I * i * x_theta))
        qbx_exp_slp += I / 4 * (hankel1(-i, k * yc) * exp(-I * i * y_theta)
                                * besselj(-i, k * xc) * exp(I * i * x_theta))
        img_qbx_exp_slp += (I / 4 * (hankel1(i, k * img_yc) *
                            exp(I * i * img_y_theta)
                            * besselj(i, k * xc) * exp(-I * i * x_theta)))
        img_qbx_exp_slp += (I / 4 * (hankel1(-i, k * img_yc) *
                            exp(-I * i * img_y_theta)
                            * besselj(-i, k * xc) * exp(I * i * x_theta)))
    qbx_exp_dlp = ny_1 * diff(qbx_exp_slp, y_1) + ny_2 * diff(qbx_exp_slp, y_2)
    exp_term_slp = lambdify([x_1, x_2, y_1, y_2,
                            nx_1, nx_2, ny_1, ny_2, r], qbx_exp_slp)
    exp_term_dlp = lambdify([x_1, x_2, y_1, y_2,
                            nx_1, nx_2, ny_1, ny_2, r], qbx_exp_dlp)
    img_qbx_exp_dlp = (ny_1 * diff(img_qbx_exp_slp, y_1) +
                       ny_2 * diff(img_qbx_exp_slp, y_2))
    img_exp_term_slp = lambdify([x_1, x_2, y_1, y_2, eta,
                                nx_1, nx_2, ny_1, ny_2, r], img_qbx_exp_slp)
    img_exp_term_dlp = lambdify([x_1, x_2, y_1, y_2, eta,
                                nx_1, nx_2, ny_1, ny_2, r], img_qbx_exp_dlp)
    return (single_layer, double_layer, exp_term_slp, exp_term_dlp,
            image_single_layer, image_double_layer, img_exp_term_slp,
            img_exp_term_dlp)


class Images_Integral:
    def __init__(self, m, beta, img_sp, img_dp):
        self.m = m
        self.beta = beta
        self.img_sp = img_sp
        self.img_dp = img_dp

    def eval_integral(self, targets, sources, source_normal_x,
                      source_normal_y):
        """
        evaluates the sum of integral of images on (m+1) dyadic intervals
        """
        dyad = 2 ** np.arange(-self.m, 1, dtype=float)
        dyadic_int = np.insert(dyad, 0, 0.0)
        npoints = 8
        ref_info = scp.legendre(npoints).weights
        ref_nodes = ref_info[:, 0]
        ref_weights = ref_info[:, 2]
        image_nodes = np.zeros((self.m+1, npoints))
        image_weights = np.zeros((self.m+1, npoints))
        for i in range(self.m + 1):
            a, b = dyadic_int[i:i+2]
            image_nodes[i] = ref_nodes * (b - a) * 0.5 + (b + a) * 0.5
            image_weights[i] = 0.5 * (b - a) * ref_weights
        image_nodes = image_nodes.reshape(-1)
        image_weights = image_weights.reshape(-1)
        # Neumann condition image
        sp_sum_int = self.img_sp(targets[0], targets[1], sources[0],
                                 sources[1], 0)
        dp_sum_int = self.img_dp(targets[0], targets[1], sources[0],
                                 sources[1], 0,
                                 source_normal_x, source_normal_y)
        for i in range((self.m + 1) * npoints):
            sp_sum_int += (2 * self.beta * 1j * self.img_sp(targets[0],
                           targets[1], sources[0], sources[1],
                           image_nodes[i]) * np.exp(1j * self.beta *
                           image_nodes[i])) * image_weights[i]
            dp_sum_int += (2 * self.beta * 1j * self.img_dp(targets[0],
                           targets[1],
                           sources[0], sources[1], image_nodes[i],
                           source_normal_x, source_normal_y) *
                           np.exp(1j * self.beta * image_nodes[i])
                           * image_weights[i])
        return sp_sum_int, dp_sum_int


def bvp(n, k, domain, alpha, qbx_exp_slp, qbx_exp_dlp, rhs, **kwargs):
    """
    solves the BVP for density
    """
    normals_x, normals_y = domain.normals.reshape(2, -1)
    nodes_x, nodes_y = domain.curve_nodes.reshape(2, -1)
    # taking exp_radius as panel_length / 2 from QBX paper
    qbx_radius = np.repeat(domain.panel_lengths, domain.npoints) / 2
    total_points = nodes_x.shape[0]
    normal_mat_x = np.broadcast_to(normals_x, (total_points, total_points))
    normal_mat_y = np.broadcast_to(normals_y, (total_points, total_points))
    node_mat_x = np.broadcast_to(nodes_x, (total_points, total_points))
    node_mat_y = np.broadcast_to(nodes_y, (total_points, total_points))
    radius_mat = np.broadcast_to(qbx_radius, (total_points, total_points)).T
    # take care of normal signs here
    D_qbx_int = qbx_exp_dlp(node_mat_x.T, node_mat_y.T, node_mat_x, node_mat_y,
                            -normal_mat_x.T, -normal_mat_y.T, normal_mat_x,
                            normal_mat_y,
                            radius_mat) * domain.curve_weights.reshape(-1)
    D_qbx_ext = qbx_exp_dlp(node_mat_x.T, node_mat_y.T, node_mat_x, node_mat_y,
                            normal_mat_x.T, normal_mat_y.T, normal_mat_x,
                            normal_mat_y,
                            radius_mat) * domain.curve_weights.reshape(-1)
    S_qbx = qbx_exp_slp(node_mat_x.T, node_mat_y.T, node_mat_x, node_mat_y,
                        normal_mat_x.T, normal_mat_y.T, normal_mat_x,
                        normal_mat_y,
                        radius_mat) * domain.curve_weights.reshape(-1)
    # averaging interior exterior limits
    rhs = rhs.reshape(-1)
    A = (D_qbx_int + D_qbx_ext) * 0.5 + 0.5 * np.identity(total_points)
    # A = D_qbx_ext
    A -= alpha * S_qbx * 1j
    # adding images and sommerfeld contribution
    if ("som_sp" in kwargs.keys()) and ("som_dp" in kwargs.keys()):
        som_sp = kwargs["som_sp"]
        som_dp = kwargs["som_dp"]
        S_som = som_sp(node_mat_x.T, node_mat_y.T, node_mat_x, node_mat_y
                       ) * domain.curve_weights.reshape(-1)
        D_som = som_dp(node_mat_x.T, node_mat_y.T, node_mat_x, node_mat_y,
                       normal_mat_x, normal_mat_y
                       ) * domain.curve_weights.reshape(-1)
        A += (D_som - alpha * 1j * S_som)
    if "imgs" in kwargs.keys():
        imgs = kwargs["imgs"]
        S_img, D_img = (imgs.eval_integral((node_mat_x.T, node_mat_y.T),
                        (node_mat_x, node_mat_y), normal_mat_x, normal_mat_y)
                        * domain.curve_weights.reshape(-1))
        A += (D_img - alpha * 1j * S_img)
    soln_density, msg = gmres(A, rhs, tol=1e-11, callback=counter)
    print("GMRES iter:", counter.niter)
    return soln_density.reshape(n, -1)


def eval_target(targets, sources, weights, source_normals, density,
                sp, dp, alpha, **kwargs):
    """
    evaluates the potential at target locations
    """
    normals_x, normals_y = source_normals.reshape(2, -1)
    nodes_x, nodes_y = sources.reshape(2, -1)
    target_number = targets.shape[1]
    total_points = nodes_x.shape[0]
    test_normal_mat_x = np.broadcast_to(normals_x,
                                        (target_number, total_points))
    test_normal_mat_y = np.broadcast_to(normals_y,
                                        (target_number, total_points))
    sources_mat_x = np.broadcast_to(nodes_x,
                                    (target_number, total_points))
    sources_mat_y = np.broadcast_to(nodes_y,
                                    (target_number, total_points))
    targets_mat_x = np.broadcast_to(targets[0],
                                    (total_points, target_number)).T
    targets_mat_y = np.broadcast_to(targets[1],
                                    (total_points, target_number)).T
    D = (dp(targets_mat_x, targets_mat_y, sources_mat_x, sources_mat_y,
         test_normal_mat_x, test_normal_mat_y) *
         weights.reshape(-1))
    S = (sp(targets_mat_x, targets_mat_y, sources_mat_x, sources_mat_y) *
         weights.reshape(-1))
    DLP_eval = (D - alpha * S*1j) @ density.reshape(-1)
    # adding images and sommerfeld contribution
    if ("som_sp" in kwargs.keys()) and ("som_dp" in kwargs.keys()):
        som_sp = kwargs["som_sp"]
        som_dp = kwargs["som_dp"]
        S_som = som_sp(targets_mat_x, targets_mat_y, sources_mat_x,
                       sources_mat_y) * weights.reshape(-1)
        D_som = som_dp(targets_mat_x, targets_mat_y, sources_mat_x,
                       sources_mat_y, test_normal_mat_x, test_normal_mat_y
                       ) * weights.reshape(-1)
        DLP_eval += (D_som - alpha * S_som * 1j) @ density.reshape(-1)
    if "imgs" in kwargs.keys():
        imgs = kwargs["imgs"]
        S_img, D_img = (imgs.eval_integral((targets_mat_x, targets_mat_y),
                        (sources_mat_x, sources_mat_y), test_normal_mat_x,
                        test_normal_mat_y) * weights.reshape(-1))
        DLP_eval += (D_img - alpha * S_img * 1j) @ density.reshape(-1)
    return DLP_eval
