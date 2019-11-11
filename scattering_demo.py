from quad import QuadratureInfo
import matplotlib.pyplot as plt
import numpy as np
# import numpy.linalg as la
from kernels import eval_sp_dp_QBX, eval_target, bvp, sommerfeld

plt.gca().set_aspect("equal")


xs = -3
ys = 0
k = 15
alpha = k  # CFIE parameter
(sp, dp, qbx_exp_slp, qbx_exp_dlp, img_sp,
 img_dp, _, _) = eval_sp_dp_QBX(4, k)
npanels = 30
domain = QuadratureInfo(npanels)

# finding the source response and solving bvp
inc = sp(domain.curve_nodes[0], domain.curve_nodes[1], xs, ys)
inc += sp(domain.curve_nodes[0], domain.curve_nodes[1], -xs, ys)
inc += sp(domain.curve_nodes[0], domain.curve_nodes[1], ys, -xs)
inc += sp(domain.curve_nodes[0], domain.curve_nodes[1], ys, xs)
rhs = -(inc)
soln1 = bvp(npanels, k, domain, alpha, qbx_exp_slp, qbx_exp_dlp, rhs)

# evaluating at target locations
tg_size = 101
x = np.linspace(-5, 5, tg_size)
y = np.linspace(-5, 5, tg_size)
X, Y = np.meshgrid(x, y)
test_targets = np.array((X.reshape(-1), Y.reshape(-1)))
exact_test = (sp(test_targets[0], test_targets[1], xs,
              ys)).reshape(tg_size, -1)
exact_test += (sp(test_targets[0], test_targets[1], -xs,
               ys)).reshape(tg_size, -1)
exact_test += (sp(test_targets[0], test_targets[1], ys,
               -xs)).reshape(tg_size, -1)
exact_test += (sp(test_targets[0], test_targets[1], ys,
               xs)).reshape(tg_size, -1)
num_test = eval_target(test_targets, domain.curve_nodes, domain.curve_weights,
                       domain.normals, soln1, sp, dp,
                       alpha).reshape(tg_size, -1)
v = np.linspace(-.15, 0.15, 400, endpoint=True)
plt.figure(1)
plt.contourf(X, Y, np.real(exact_test), v, cmap='twilight')
plt.colorbar()
plt.figure(2)
plt.contourf(X, Y, np.real(exact_test + num_test), v,
             cmap='twilight')
plt.fill(domain.curve_nodes[0], domain.curve_nodes[1], 'w')
plt.gca().set_aspect("equal")
plt.xticks([])
plt.yticks([])
plt.savefig('try.png', format='png', dpi=300,
            bbox_inches='tight', pad_inches=0)
# plt.colorbar()
plt.show()
