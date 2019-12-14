import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from kernels import sommerfeld
from plotset import plotset


def som_convg(k, beta, ys):
    """
    checks convergence of full Sommerfeld integral
    """
    plotset()
    intervals = np.arange(2, 12, 2)
    xs = 0
    test_targets = np.array([[4, 1], [2, 1], [2, 0.1]]).T
    som_err = []
    # finest soln
    # som_sp, _ = sommerfeld(k, beta, 12, "full")
    som_sp, _ = sommerfeld(k, beta, 12, "far")
    som_fine = som_sp(test_targets[0], test_targets[1], xs, ys)
    som_fine_norm = la.norm(som_fine)
    for i in intervals:
        # som_sp, _ = sommerfeld(k, beta, i, "full")
        som_sp, _ = sommerfeld(k, beta, i, "far")
        som_test = som_sp(test_targets[0], test_targets[1], xs, ys)
        err = la.norm(som_test - som_fine) / som_fine_norm
        som_err.append(err)

    plt.semilogy(intervals, np.array(som_err), "-o", label="y = " + str(ys))


if __name__ == "__main__":
    som_convg(10, 2, 1)
    som_convg(10, 2, 0.5)
    som_convg(10, 2, 0.1)
    title = "Sommerfeld integral convergence - far field"
    plt.xlabel("$\lambda(t)$", size=25)
    plt.ylabel("sommerfeld integral", size=25)
    plt.title(title, size=25)
    plt.legend(loc="lower left", ncol=1, frameon=True)
    plt.show()
