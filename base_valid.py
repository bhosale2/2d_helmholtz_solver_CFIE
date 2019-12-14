from quad import QuadratureInfo
import matplotlib.pyplot as plt
import numpy as np

# import numpy.linalg as la
from kernels import eval_sp_dp_QBX, eval_target, bvp

plt.gca().set_aspect("equal")


def main():
    k = 10
    alpha = k  # CFIE parameter
    sp, dp, qbx_exp_slp, qbx_exp_dlp, _, _, _, _ = eval_sp_dp_QBX(4, k)
    npanels = 30
    domain = QuadratureInfo(npanels, show_bound=1)
    xs = 0.75
    ys = 1.75

    tg_size = 101
    x = np.linspace(-2, 5, tg_size)
    y = np.linspace(-2, 5, tg_size)
    X, Y = np.meshgrid(x, y)
    test_targets = np.array((X.reshape(-1), Y.reshape(-1)))
    rhs = sp(domain.curve_nodes[0], domain.curve_nodes[1], xs, ys)
    soln1 = bvp(npanels, k, domain, alpha, qbx_exp_slp, qbx_exp_dlp, rhs)
    num_test = eval_target(
        test_targets,
        domain.curve_nodes,
        domain.curve_weights,
        domain.normals,
        soln1,
        sp,
        dp,
        alpha,
    ).reshape(tg_size, -1)
    exact_test = sp(test_targets[0], test_targets[1], xs, ys).reshape(tg_size, -1)
    err = np.abs(num_test - exact_test) / np.abs(exact_test)
    # plt.contourf(X, Y, np.real(exact_test), cmap='magma', levels=100)
    v = np.linspace(-0.15, 0.15, 100, endpoint=True)
    plt.figure(1)
    plt.contourf(X, Y, np.real(num_test), v, cmap="twilight")
    plt.plot(xs, ys, "ro")
    plt.fill(domain.curve_nodes[0], domain.curve_nodes[1], "w")
    show_targets = np.array([[-1, 0], [4, 0], [0, -1], [0, 4]]).T
    plt.plot(show_targets[0], show_targets[1], "go")
    plt.xlabel("X", size=10)
    plt.ylabel("Y", size=10)
    plt.colorbar()
    plt.figure(2)
    plt.contourf(X, Y, np.log10(err), cmap="magma", levels=200)
    plt.fill(domain.curve_nodes[0], domain.curve_nodes[1], "w")
    plt.plot(xs, ys, "ro")
    plt.plot(show_targets[0], show_targets[1], "go")
    plt.xlabel("X", size=10)
    plt.ylabel("Y", size=10)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
