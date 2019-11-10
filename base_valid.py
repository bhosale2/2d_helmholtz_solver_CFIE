from quad import QuadratureInfo
import matplotlib.pyplot as plt
import numpy as np
# import numpy.linalg as la
from kernels import eval_sp_dp_QBX, eval_target, bvp
from quad import delta

plt.gca().set_aspect("equal")


k = 10
alpha = 10  # CFIE parameter
sp, dp, qbx_exp_slp, qbx_exp_dlp, _, _, _, _ = eval_sp_dp_QBX(4, k)
npanels = 20
domain = QuadratureInfo(npanels)

tg_size = 101
x = np.linspace(-2, 5, tg_size)
y = np.linspace(-2, 5, tg_size)
X, Y = np.meshgrid(x, y)
test_targets = np.array((X.reshape(-1), Y.reshape(-1)))
rhs = sp(domain.curve_nodes[0], domain.curve_nodes[1], 1.1, 1.1 +
         delta)
soln1 = bvp(npanels, k, domain, alpha, qbx_exp_slp, qbx_exp_dlp,
            rhs)
num_test = eval_target(test_targets, domain.curve_nodes, domain.curve_weights,
                       domain.normals, soln1, sp, dp,
                       alpha).reshape(tg_size, -1)
exact_test = sp(test_targets[0], test_targets[1], 1.1, 1.1 +
                delta).reshape(tg_size, -1)
err = np.abs(num_test - exact_test)
# plt.contourf(X, Y, np.real(exact_test), cmap='magma', levels=100)
v = np.linspace(-.15, 0.15, 100, endpoint=True)
plt.figure(1)
plt.contourf(X, Y, np.real(num_test), v,
             cmap='twilight')
plt.fill(domain.curve_nodes[0], domain.curve_nodes[1], 'w')
plt.colorbar()
plt.figure(2)
plt.contourf(X, Y, np.log(err), cmap='magma', levels=200)
plt.fill(domain.curve_nodes[0], domain.curve_nodes[1], 'w')
plt.colorbar()
plt.show()
