from quad import QuadratureInfo
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from kernels import eval_sp_dp_QBX, eval_target, bvp
from plotset import plotset
from scipy import stats


def ext_helmholtz_convg(qbx_order, k):
    """
    Tests external Helmholtz solution by placing monopoles inside domain
    """
    alpha = k  # CFIE parameter
    sp, dp, qbx_exp_slp, qbx_exp_dlp, _, _, _, _ = eval_sp_dp_QBX(qbx_order, k)

    xs = 0.75
    ys = 1.75
    test_targets = np.array([[-1, 0], [4, 0], [0, -1], [0, 4]]).T
    # test_targets = np.array([[-5, 0], [5, 0], [0, -5], [0, 5]]).T
    exact_test = sp(test_targets[0], test_targets[1], xs, ys)
    plotset()
    n_panels = np.arange(10, 60, 5)
    convg_err = []
    for n in n_panels:
        domain = QuadratureInfo(n, show_bound=0)
        rhs = sp(domain.curve_nodes[0], domain.curve_nodes[1], xs, ys)
        soln = bvp(n, k, domain, alpha, qbx_exp_slp, qbx_exp_dlp, rhs)
        num_test = eval_target(
            test_targets,
            domain.curve_nodes,
            domain.curve_weights,
            domain.normals,
            soln,
            sp,
            dp,
            alpha,
        )
        er = la.norm(np.abs(num_test - exact_test) / la.norm(np.abs(exact_test)))
        convg_err.append(er)

    plt.figure(1)
    plt.loglog(n_panels, np.array(convg_err), "-o", label="QBX" + str(qbx_order))
    plt.loglog(
        n_panels,
        5e4 * np.power(n_panels.astype(float), -qbx_order - 1),
        label="$N^{-5}$",
    )
    plt.loglog(
        n_panels, 2e3 * np.power(n_panels.astype(float), -qbx_order), label="$N^{-4}$"
    )
    title = "Helmholtz exterior problem: target evaluation convergence"
    plt.xlabel("Number of panels", size=25)
    plt.ylabel("$|u - u_e| / |u_e|$", size=25)
    plt.title(title, size=25)
    plt.legend(loc="upper right", ncol=1, frameon=True)
    plt.show()
    slope, _, _, _, _ = stats.linregress(np.log(n_panels), np.log(np.array(convg_err)))
    print(slope)


if __name__ == "__main__":
    ext_helmholtz_convg(4, 10)
